"""Train the confirmed H03 FULL recipe. Run with single-process torchrun."""
import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rotate, InterpolationMode

from local_semi_role_la import BreakBias_LA_Role
from model.UNET3D import UNet3D
from model.VNet import VNetv2 as VNet
from flow.model import RNVP
from flow.distributions import FastGMM
from flow.loss import FlowLoss
from util.losses import DiceLoss
from protocol_support import protocol_joint_validation, record_snapshot

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

def ddfp_perturb_feature_asymmetric(features, nf_model, loss_fn, eps, direction='descending'):
    b, c, d, h, w = features.size()
    features_reshaped = features.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
    features_reshaped.requires_grad_()
    z, log_jac_det = nf_model(features_reshaped)
    nf_loss, _, _ = loss_fn(z, sldj=log_jac_det)
    grad = torch.autograd.grad(nf_loss, features_reshaped, retain_graph=False)[0]
    grad_norm = torch.norm(grad, p=2, dim=1).view(-1, 1) + 1e-08
    if direction == 'descending':
        delta = eps * grad / grad_norm
    elif direction == 'ascending':
        delta = -eps * grad / grad_norm
    else:
        raise ValueError("Direction must be 'descending' or 'ascending'")
    perturbed_features_reshaped = (features_reshaped + delta).detach()
    perturbed_features = perturbed_features_reshaped.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
    return perturbed_features


def main():
    parser = argparse.ArgumentParser(description='BreakBias H03 FULL, LA V10')
    parser.add_argument('--config', default='configs/la5_h03_full.json')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='outputs/la5_h03_full')
    parser.add_argument('--smoke', action='store_true', help='Two updates with ADP active and one validation case; not a scientific run.')
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = json.loads(Path(cli.config).read_text())
    assert cfg['dataset'] == 'LA' and cfg['model'] == 'UV'
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1, 'This recipe was run with one DDP process.'
    cfg.update(data_root=str(Path(cli.data_root).resolve()), output_dir=str(Path(cli.output_dir).resolve()))
    cfg['smoke'] = cli.smoke
    if cli.smoke:
        cfg['max_iterations'] = 2
        cfg['ddfp']['start_epoch'] = 0
    cfg['val_list'] = str(root / 'partitions/LA/val.list')
    snapshot_path = cfg['output_dir']
    Path(snapshot_path).mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(ddfp_dir1='ascending', ddfp_loss_weight=cfg['ddfp']['loss_weight'],
                           cons_weight=cfg['cons_weight'], rampup_iter=cfg['rampup_iter'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    init_seeds(cfg['seed'])
    n_filters1, n_filters2 = cfg['n_filters1'], cfg['n_filters2']
    backbone = cfg['model']
    def build_model(arch, n_filters):
        if arch == 'U':
            return (UNet3D(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=n_filters, normalization='batchnorm', has_dropout=True), n_filters * 8)
        if arch == 'V':
            return (VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=n_filters, normalization='batchnorm', has_dropout=True), n_filters * 16)
        raise ValueError(f'Unknown backbone arch: {arch}')
    model1, last_feature_dim1 = build_model(backbone[0], n_filters1)
    model2, last_feature_dim2 = build_model(backbone[1], n_filters2)
    teacher1, _ = build_model(backbone[0], n_filters1)
    teacher2, _ = build_model(backbone[1], n_filters2)
    cfg_ddfp = cfg.get('ddfp', {})
    cfg_flow = {'num_blocks': cfg_ddfp.get('num_blocks', 8), 'layer_dims': cfg_ddfp.get('layer_dims', 512), 'n_components': cfg['nclass'], 'clamping': cfg_ddfp.get('clamping', 2.0), 'noise': cfg_ddfp.get('noise', 1e-05), 'grad_clip': cfg_ddfp.get('grad_clip', 1.0), 'lr': cfg_ddfp.get('lr', 0.0001), 'start_epoch': cfg_ddfp.get('start_epoch', 10), 'sample_batch_size': cfg_ddfp.get('sample_batch_size', 20480), 'loss_weight': cfg_ddfp.get('loss_weight', 0.5), 'ema_decay': cfg_ddfp.get('ema_decay', 0.999), 'eps': cfg_ddfp.get('eps', 0.01)}
    means1 = torch.randn(cfg_flow['n_components'], last_feature_dim1)
    nf_model1 = RNVP({'input_dims': last_feature_dim1, **cfg_flow}, means1, learnable_mean=True)
    means2 = torch.randn(cfg_flow['n_components'], last_feature_dim2)
    nf_model2 = RNVP({'input_dims': last_feature_dim2, **cfg_flow}, means2, learnable_mean=True)
    model1 = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    model2 = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    teacher1 = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher1).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    teacher2 = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher2).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    nf_model1 = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(nf_model1).cuda(), device_ids=[local_rank], find_unused_parameters=True, static_graph=True)
    nf_model2 = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(nf_model2).cuda(), device_ids=[local_rank], find_unused_parameters=True, static_graph=True)
    optimizer1 = SGD(model1.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)
    optimizer2 = SGD(model2.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)
    optimizer_nf1 = Adam(nf_model1.parameters(), lr=cfg_flow['lr'], betas=(0.5, 0.9))
    optimizer_nf2 = Adam(nf_model2.parameters(), lr=cfg_flow['lr'], betas=(0.5, 0.9))
    for p in teacher1.parameters():
        p.requires_grad = False
    for p in teacher2.parameters():
        p.requires_grad = False
    ce_loss = nn.CrossEntropyLoss(ignore_index=255).cuda()
    dice_loss = DiceLoss(cfg['nclass']).cuda()
    prior1 = FastGMM(nf_model1.module.means)
    loss_flow_fn1 = FlowLoss(prior1)
    prior2 = FastGMM(nf_model2.module.means)
    loss_flow_fn2 = FlowLoss(prior2)
    labeled_id_path = root / f"partitions/LA/1_{cfg['labeled_rate']}/labeled.list"
    unlabeled_id_path = root / f"partitions/LA/1_{cfg['labeled_rate']}/unlabeled.list"
    dataset_kwargs = dict(tree_mode=cfg['tree_mode'], tree_areathrehold=cfg['tree_areathrehold'],
                          tree_topoprob=cfg['tree_topoprob'], role_trigger_prob=cfg['role_trigger_prob'],
                          role_altitude_jitter=cfg['role_altitude_jitter'])
    trainset_u = BreakBias_LA_Role('LA', cfg['data_root'], 'train_u', cfg['crop_size'], unlabeled_id_path, **dataset_kwargs)
    trainset_l = BreakBias_LA_Role('LA', cfg['data_root'], 'train_l', cfg['crop_size'], labeled_id_path, nsample=len(trainset_u.ids), **dataset_kwargs)
    valset = BreakBias_LA_Role('LA', cfg['data_root'], 'val', id_path=cfg['val_list'])
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2, drop_last=False, sampler=valsampler)
    max_iterations = cfg['max_iterations']
    accumulation_steps = cfg['accumulation_steps']
    iters_per_epoch = len(trainloader_l) // accumulation_steps
    assert iters_per_epoch > 0
    max_epoch = max_iterations // iters_per_epoch + 1
    ddfp_start_epoch = cfg_flow['start_epoch']
    iter_num = 0
    with torch.no_grad():
        for t_p, s_p in zip(teacher1.parameters(), model1.parameters()):
            t_p.data.copy_(s_p.data)
        for t_p, s_p in zip(teacher2.parameters(), model2.parameters()):
            t_p.data.copy_(s_p.data)
    (Path(snapshot_path) / 'effective_config.json').write_text(json.dumps(cfg, indent=2))
    with (Path(snapshot_path) / 'train.csv').open('w', newline='') as log:
        fields = ['iter', 'epoch', 'loss', 'supervised1', 'supervised2', 'cps1', 'cps2', 'ddfp1', 'ddfp2', 'consistency_weight', 'ddfp_active']
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        for epoch in range(max_epoch):
            trainsampler_l.set_epoch(epoch)
            trainsampler_u.set_epoch(epoch)
            loader = zip(trainloader_l, trainloader_u)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer_nf1.zero_grad()
            optimizer_nf2.zero_grad()
            for i, (labeled_sampled, unlabeled_sampled) in enumerate(loader):
                labeled_sampled_similar, labeled_sampled_dissimilar, labeled_label_batch, rotate_labeled = (labeled_sampled['image_similar'], labeled_sampled['image_dissimilar'], labeled_sampled['label'], labeled_sampled['rotate_angle'])
                unlabeled_sampled_similar, unlabeled_sampled_dissimilar, unlabeled_weak_batch, rotate_unlabeled = (unlabeled_sampled['image_similar'], unlabeled_sampled['image_dissimilar'], unlabeled_sampled['image_weak'], unlabeled_sampled['rotate_angle'])
                labeled_sampled_similar, labeled_sampled_dissimilar, labeled_label_batch = (labeled_sampled_similar.cuda(), labeled_sampled_dissimilar.cuda(), labeled_label_batch.cuda())
                unlabeled_sampled_similar, unlabeled_sampled_dissimilar, unlabeled_weak_batch = (unlabeled_sampled_similar.cuda(), unlabeled_sampled_dissimilar.cuda(), unlabeled_weak_batch.cuda())
                model1.train()
                model2.train()
                if epoch >= ddfp_start_epoch:
                    nf_model1.train()
                    with torch.no_grad():
                        v_ema_dict1 = teacher1(unlabeled_weak_batch)
                        v_ema1 = v_ema_dict1['features'][-1].detach()
                    b_nf, c_nf, d_nf, h_nf, w_nf = v_ema1.size()
                    v_ema_reshaped1 = v_ema1.permute(0, 2, 3, 4, 1).contiguous().view(-1, c_nf)
                    sample_num = min(cfg_flow['sample_batch_size'], v_ema_reshaped1.shape[0])
                    sample_idx = np.random.choice(v_ema_reshaped1.shape[0], sample_num, False)
                    input_nf_sample1 = v_ema_reshaped1[sample_idx] + cfg_flow['noise'] * torch.randn_like(v_ema_reshaped1[sample_idx])
                    z1, log_jac_det1 = nf_model1(input_nf_sample1)
                    nf_loss1, _, _ = loss_flow_fn1(z1, sldj=log_jac_det1)
                    (nf_loss1 / accumulation_steps).backward()
                    nf_model2.train()
                    with torch.no_grad():
                        v_ema_dict2 = teacher2(unlabeled_weak_batch)
                        v_ema2 = v_ema_dict2['features'][-1].detach()
                    b_nf, c_nf, d_nf, h_nf, w_nf = v_ema2.size()
                    v_ema_reshaped2 = v_ema2.permute(0, 2, 3, 4, 1).contiguous().view(-1, c_nf)
                    sample_num = min(cfg_flow['sample_batch_size'], v_ema_reshaped2.shape[0])
                    sample_idx = np.random.choice(v_ema_reshaped2.shape[0], sample_num, False)
                    input_nf_sample2 = v_ema_reshaped2[sample_idx] + cfg_flow['noise'] * torch.randn_like(v_ema_reshaped2[sample_idx])
                    z2, log_jac_det2 = nf_model2(input_nf_sample2)
                    nf_loss2, _, _ = loss_flow_fn2(z2, sldj=log_jac_det2)
                    (nf_loss2 / accumulation_steps).backward()
                    nf_model1.eval()
                    nf_model2.eval()
                out_dict1_l = model1(labeled_sampled_similar)
                labeled_output1 = out_dict1_l['pred']
                labeled_output1_soft = torch.softmax(labeled_output1, dim=1)
                out_dict2_l = model2(labeled_sampled_dissimilar)
                labeled_output2_raw = out_dict2_l['pred']
                labeled_output2 = torch.zeros_like(labeled_output2_raw)
                for batch_index in range(labeled_output2.shape[0]):
                    labeled_output2[batch_index] = rotate(labeled_output2_raw[batch_index], -int(rotate_labeled[batch_index].item()), interpolation=InterpolationMode.NEAREST)
                labeled_output2_soft = torch.softmax(labeled_output2, dim=1)
                loss1 = 0.5 * (ce_loss(labeled_output1, labeled_label_batch.long()) + dice_loss(labeled_output1_soft, labeled_label_batch.unsqueeze(1)))
                loss2 = 0.5 * (ce_loss(labeled_output2, labeled_label_batch.long()) + dice_loss(labeled_output2_soft, labeled_label_batch.unsqueeze(1)))
                supervised_loss_total = loss1 + loss2
                (supervised_loss_total / accumulation_steps).backward()
                out_dict1_u = model1(unlabeled_sampled_similar)
                unlabeled_output1, features1_all = (out_dict1_u['pred'], out_dict1_u['features'])
                out_dict2_u = model2(unlabeled_sampled_dissimilar)
                unlabeled_output2_raw, features2_all = (out_dict2_u['pred'], out_dict2_u['features'])
                unlabeled_output2 = torch.zeros_like(unlabeled_output2_raw)
                for batch_index in range(unlabeled_output2.shape[0]):
                    unlabeled_output2[batch_index] = rotate(unlabeled_output2_raw[batch_index], -int(rotate_unlabeled[batch_index].item()), interpolation=InterpolationMode.NEAREST)
                pseudo_unlabeled_output1 = torch.argmax(torch.softmax(unlabeled_output1.detach(), dim=1), dim=1)
                pseudo_unlabeled_output2 = torch.argmax(torch.softmax(unlabeled_output2.detach(), dim=1), dim=1)
                pseudo_supervision1_orig = ce_loss(unlabeled_output1, pseudo_unlabeled_output2)
                pseudo_supervision2_orig = ce_loss(unlabeled_output2, pseudo_unlabeled_output1)
                pseudo_supervision1_ddfp = torch.tensor(0.0).cuda(local_rank)
                pseudo_supervision2_ddfp = torch.tensor(0.0).cuda(local_rank)
                if epoch >= ddfp_start_epoch:
                    unlabeled_fts1 = features1_all[-1]
                    v1_ddfp = ddfp_perturb_feature_asymmetric(unlabeled_fts1, nf_model1, loss_flow_fn1, cfg_flow['eps'], direction=args.ddfp_dir1)
                    p1_ddfp_features = features1_all[:-1] + [v1_ddfp]
                    p1_ddfp = model1.module.decoder(p1_ddfp_features)
                    pseudo_supervision1_ddfp = ce_loss(p1_ddfp, pseudo_unlabeled_output2)
                    unlabeled_fts2 = features2_all[-1]
                    v2_ddfp = ddfp_perturb_feature_asymmetric(unlabeled_fts2, nf_model2, loss_flow_fn2, cfg_flow['eps'], direction='descending')
                    p2_ddfp_features = features2_all[:-1] + [v2_ddfp]
                    p2_ddfp_raw = model2.module.decoder(p2_ddfp_features)
                    p2_ddfp = torch.zeros_like(p2_ddfp_raw)
                    for batch_index in range(p2_ddfp.shape[0]):
                        p2_ddfp[batch_index] = rotate(p2_ddfp_raw[batch_index], -int(rotate_unlabeled[batch_index].item()), interpolation=InterpolationMode.NEAREST)
                    pseudo_supervision2_ddfp = ce_loss(p2_ddfp, pseudo_unlabeled_output1)
                consistency_weight = sigmoid_rampup(iter_num, args.rampup_iter) * args.cons_weight
                lambda_ddfp = args.ddfp_loss_weight
                unsupervised_loss1 = (1.0 - lambda_ddfp) * pseudo_supervision1_orig + lambda_ddfp * pseudo_supervision1_ddfp
                unsupervised_loss2 = (1.0 - lambda_ddfp) * pseudo_supervision2_orig + lambda_ddfp * pseudo_supervision2_ddfp
                model1_loss = loss1 + consistency_weight * unsupervised_loss1
                model2_loss = loss2 + consistency_weight * unsupervised_loss2
                loss = model1_loss + model2_loss
                unsupervised_loss_total = consistency_weight * (unsupervised_loss1 + unsupervised_loss2)
                (unsupervised_loss_total / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    if cfg_flow['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(model1.parameters(), cfg_flow['grad_clip'])
                        torch.nn.utils.clip_grad_norm_(model2.parameters(), cfg_flow['grad_clip'])
                        if epoch >= ddfp_start_epoch:
                            torch.nn.utils.clip_grad_norm_(nf_model1.parameters(), cfg_flow['grad_clip'])
                            torch.nn.utils.clip_grad_norm_(nf_model2.parameters(), cfg_flow['grad_clip'])
                    optimizer1.step()
                    optimizer2.step()
                    if epoch >= ddfp_start_epoch:
                        optimizer_nf1.step()
                        optimizer_nf2.step()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    optimizer_nf1.zero_grad()
                    optimizer_nf2.zero_grad()
                    iter_num += 1
                    with torch.no_grad():
                        ema_decay = cfg_flow['ema_decay']
                        for t_params, s_params in zip(teacher1.parameters(), model1.parameters()):
                            t_params.data.copy_(ema_decay * t_params.data + (1.0 - ema_decay) * s_params.data)
                        for t_params, s_params in zip(teacher2.parameters(), model2.parameters()):
                            t_params.data.copy_(ema_decay * t_params.data + (1.0 - ema_decay) * s_params.data)
                    lr_ = cfg['lr'] * (1 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer1.param_groups:
                        param_group['lr'] = lr_
                    for param_group in optimizer2.param_groups:
                        param_group['lr'] = lr_
                    should_validate = iter_num > 0 and iter_num % cfg['validation_interval'] == 0
                    if (i + 1) % accumulation_steps == 0 and iter_num > 0 and (iter_num % cfg['validation_interval'] == 0):
                        cfg['evaluation_iter'] = iter_num
                        model1.eval()
                        model2.eval()
                        avg_dice1, per_class_dice1, avg_dice2, per_class_dice2, bias_metrics = protocol_joint_validation(model1, model2, valloader, cfg, stride_xy=18, stride_z=4, patch_size=cfg['crop_size'], tau=args.bias_tau)
                if (i + 1) % accumulation_steps == 0:
                    values = [loss, loss1, loss2, pseudo_supervision1_orig, pseudo_supervision2_orig,
                              pseudo_supervision1_ddfp, pseudo_supervision2_ddfp]
                    if not all(torch.isfinite(v).all().item() for v in values):
                        raise FloatingPointError(f'Nonfinite loss at update {iter_num}')
                    writer.writerow(dict(zip(fields, [iter_num, epoch, *[v.item() for v in values], consistency_weight, epoch >= ddfp_start_epoch])))
                    log.flush()
                    print(f'update={iter_num}/{max_iterations} loss={loss.item():.6f}', flush=True)
                    if iter_num % cfg['validation_interval'] == 0 or iter_num == max_iterations:
                        cfg['evaluation_iter'] = iter_num
                        model1.eval()
                        model2.eval()
                        a, _, b, _, _ = protocol_joint_validation(model1, model2, valloader, cfg)
                        checkpoint = dict(epoch=epoch, iter_num=iter_num, protocol_config=cfg,
                            model1_state_dict=model1.module.state_dict(), model2_state_dict=model2.module.state_dict(),
                            teacher1_state_dict=teacher1.module.state_dict(), teacher2_state_dict=teacher2.module.state_dict(),
                            nf_model1_state_dict=nf_model1.module.state_dict(), nf_model2_state_dict=nf_model2.module.state_dict(),
                            optimizer1_state_dict=optimizer1.state_dict(), optimizer2_state_dict=optimizer2.state_dict(),
                            optimizer_nf1_state_dict=optimizer_nf1.state_dict(), optimizer_nf2_state_dict=optimizer_nf2.state_dict())
                        record_snapshot(checkpoint, snapshot_path, a, b, cfg)
                        print(f'validation update={iter_num} model1={a:.9f} model2={b:.9f}', flush=True)
                if iter_num >= max_iterations:
                    break
            if iter_num >= max_iterations:
                break
    marker = 'SMOKE_COMPLETE.json' if cfg['smoke'] else 'TRAIN_COMPLETE.json'
    (Path(snapshot_path) / marker).write_text(json.dumps(dict(iterations=iter_num, seed=cfg['seed'], smoke=cfg['smoke'])))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
