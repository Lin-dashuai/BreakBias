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
from dataset.la import BreakBiasLADataset
from adp import adp_perturb_feature_asymmetric
from util.checkpoints import normalize_config
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

def main():
    parser = argparse.ArgumentParser(description='BreakBias H03 FULL, LA V10')
    parser.add_argument('--config', default='configs/la5_h03_full.json')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='outputs/la5_h03_full')
    parser.add_argument('--smoke', action='store_true', help='Two updates with ADP active and one validation case; not a scientific run.')
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = normalize_config(json.loads(Path(cli.config).read_text()))
    assert cfg['dataset'] == 'LA' and cfg['model'] == 'UV'
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1, 'This recipe was run with one DDP process.'
    cfg.update(data_root=str(Path(cli.data_root).resolve()), output_dir=str(Path(cli.output_dir).resolve()))
    cfg['smoke'] = cli.smoke
    if cli.smoke:
        cfg['max_iterations'] = 2
        cfg['adp']['start_epoch'] = 0
    cfg['val_list'] = str(root / 'partitions/LA/val.list')
    snapshot_path = cfg['output_dir']
    Path(snapshot_path).mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(adp_direction_exploiter='ascending', adp_loss_weight=cfg['adp']['loss_weight'], consistency_weight_max=cfg['consistency_weight_max'], consistency_ramp_iters=cfg['consistency_ramp_iters'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    init_seeds(cfg['seed'])
    exploiter_filters, explorer_filters = (cfg['exploiter_filters'], cfg['explorer_filters'])
    backbone = cfg['model']

    def build_model(arch, n_filters):
        if arch == 'U':
            return (UNet3D(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=n_filters, normalization='batchnorm', has_dropout=True), n_filters * 8)
        if arch == 'V':
            return (VNet(n_channels=cfg['image_channel'], n_classes=cfg['nclass'], n_filters=n_filters, normalization='batchnorm', has_dropout=True), n_filters * 16)
        raise ValueError(f'Unknown backbone arch: {arch}')
    exploiter, exploiter_feature_dim = build_model(backbone[0], exploiter_filters)
    explorer, explorer_feature_dim = build_model(backbone[1], explorer_filters)
    exploiter_teacher, _ = build_model(backbone[0], exploiter_filters)
    explorer_teacher, _ = build_model(backbone[1], explorer_filters)
    cfg_adp = cfg.get('adp', {})
    cfg_flow = {'num_blocks': cfg_adp.get('num_blocks', 8), 'layer_dims': cfg_adp.get('layer_dims', 512), 'n_components': cfg['nclass'], 'clamping': cfg_adp.get('clamping', 2.0), 'noise': cfg_adp.get('noise', 1e-05), 'grad_clip': cfg_adp.get('grad_clip', 1.0), 'lr': cfg_adp.get('lr', 0.0001), 'start_epoch': cfg_adp.get('start_epoch', 10), 'sample_batch_size': cfg_adp.get('sample_batch_size', 20480), 'loss_weight': cfg_adp.get('loss_weight', 0.5), 'ema_decay': cfg_adp.get('ema_decay', 0.999), 'eps': cfg_adp.get('eps', 0.01)}
    exploiter_means = torch.randn(cfg_flow['n_components'], exploiter_feature_dim)
    exploiter_density = RNVP({'input_dims': exploiter_feature_dim, **cfg_flow}, exploiter_means, learnable_mean=True)
    explorer_means = torch.randn(cfg_flow['n_components'], explorer_feature_dim)
    explorer_density = RNVP({'input_dims': explorer_feature_dim, **cfg_flow}, explorer_means, learnable_mean=True)
    exploiter = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(exploiter).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    explorer = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(explorer).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    exploiter_teacher = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(exploiter_teacher).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    explorer_teacher = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(explorer_teacher).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    exploiter_density = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(exploiter_density).cuda(), device_ids=[local_rank], find_unused_parameters=True, static_graph=True)
    explorer_density = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(explorer_density).cuda(), device_ids=[local_rank], find_unused_parameters=True, static_graph=True)
    exploiter_optimizer = SGD(exploiter.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)
    explorer_optimizer = SGD(explorer.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)
    exploiter_density_optimizer = Adam(exploiter_density.parameters(), lr=cfg_flow['lr'], betas=(0.5, 0.9))
    explorer_density_optimizer = Adam(explorer_density.parameters(), lr=cfg_flow['lr'], betas=(0.5, 0.9))
    for p in exploiter_teacher.parameters():
        p.requires_grad = False
    for p in explorer_teacher.parameters():
        p.requires_grad = False
    ce_loss = nn.CrossEntropyLoss(ignore_index=255).cuda()
    dice_loss = DiceLoss(cfg['nclass']).cuda()
    exploiter_prior = FastGMM(exploiter_density.module.means)
    exploiter_density_loss_fn = FlowLoss(exploiter_prior)
    explorer_prior = FastGMM(explorer_density.module.means)
    explorer_density_loss_fn = FlowLoss(explorer_prior)
    labeled_id_path = root / f"partitions/LA/1_{cfg['labeled_rate']}/labeled.list"
    unlabeled_id_path = root / f"partitions/LA/1_{cfg['labeled_rate']}/unlabeled.list"
    dataset_kwargs = dict(rvp_enabled=cfg['rvp_enabled'], rvp_prob=cfg['rvp_prob'], rvp_altitude_jitter=cfg['rvp_altitude_jitter'])
    trainset_u = BreakBiasLADataset('LA', cfg['data_root'], 'train_u', cfg['crop_size'], unlabeled_id_path, **dataset_kwargs)
    trainset_l = BreakBiasLADataset('LA', cfg['data_root'], 'train_l', cfg['crop_size'], labeled_id_path, nsample=len(trainset_u.ids), **dataset_kwargs)
    valset = BreakBiasLADataset('LA', cfg['data_root'], 'val', id_path=cfg['val_list'])
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
    adp_start_epoch = cfg_flow['start_epoch']
    iter_num = 0
    with torch.no_grad():
        for t_p, s_p in zip(exploiter_teacher.parameters(), exploiter.parameters()):
            t_p.data.copy_(s_p.data)
        for t_p, s_p in zip(explorer_teacher.parameters(), explorer.parameters()):
            t_p.data.copy_(s_p.data)
    (Path(snapshot_path) / 'effective_config.json').write_text(json.dumps(cfg, indent=2))
    with (Path(snapshot_path) / 'train.csv').open('w', newline='') as log:
        fields = ['iter', 'epoch', 'loss', 'supervised_exploiter', 'supervised_explorer', 'cps_exploiter', 'cps_explorer', 'adp_exploiter', 'adp_explorer', 'consistency_weight', 'adp_active']
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        for epoch in range(max_epoch):
            trainsampler_l.set_epoch(epoch)
            trainsampler_u.set_epoch(epoch)
            loader = zip(trainloader_l, trainloader_u)
            exploiter_optimizer.zero_grad()
            explorer_optimizer.zero_grad()
            exploiter_density_optimizer.zero_grad()
            explorer_density_optimizer.zero_grad()
            for i, (labeled_sampled, unlabeled_sampled) in enumerate(loader):
                labeled_ap, labeled_be, labeled_label_batch, rotate_labeled = (labeled_sampled['image_ap'], labeled_sampled['image_be'], labeled_sampled['label'], labeled_sampled['rotate_angle'])
                unlabeled_ap, unlabeled_be, unlabeled_weak_batch, rotate_unlabeled = (unlabeled_sampled['image_ap'], unlabeled_sampled['image_be'], unlabeled_sampled['image_weak'], unlabeled_sampled['rotate_angle'])
                labeled_ap, labeled_be, labeled_label_batch = (labeled_ap.cuda(), labeled_be.cuda(), labeled_label_batch.cuda())
                unlabeled_ap, unlabeled_be, unlabeled_weak_batch = (unlabeled_ap.cuda(), unlabeled_be.cuda(), unlabeled_weak_batch.cuda())
                exploiter.train()
                explorer.train()
                if epoch >= adp_start_epoch:
                    exploiter_density.train()
                    with torch.no_grad():
                        teacher_outputs_exploiter = exploiter_teacher(unlabeled_weak_batch)
                        teacher_features_exploiter = teacher_outputs_exploiter['features'][-1].detach()
                    b_nf, c_nf, d_nf, h_nf, w_nf = teacher_features_exploiter.size()
                    flat_teacher_features_exploiter = teacher_features_exploiter.permute(0, 2, 3, 4, 1).contiguous().view(-1, c_nf)
                    sample_num = min(cfg_flow['sample_batch_size'], flat_teacher_features_exploiter.shape[0])
                    sample_idx = np.random.choice(flat_teacher_features_exploiter.shape[0], sample_num, False)
                    density_samples_exploiter = flat_teacher_features_exploiter[sample_idx] + cfg_flow['noise'] * torch.randn_like(flat_teacher_features_exploiter[sample_idx])
                    z_exploiter, log_jacobian_exploiter = exploiter_density(density_samples_exploiter)
                    density_loss_exploiter, _, _ = exploiter_density_loss_fn(z_exploiter, sldj=log_jacobian_exploiter)
                    (density_loss_exploiter / accumulation_steps).backward()
                    explorer_density.train()
                    with torch.no_grad():
                        teacher_outputs_explorer = explorer_teacher(unlabeled_weak_batch)
                        teacher_features_explorer = teacher_outputs_explorer['features'][-1].detach()
                    b_nf, c_nf, d_nf, h_nf, w_nf = teacher_features_explorer.size()
                    flat_teacher_features_explorer = teacher_features_explorer.permute(0, 2, 3, 4, 1).contiguous().view(-1, c_nf)
                    sample_num = min(cfg_flow['sample_batch_size'], flat_teacher_features_explorer.shape[0])
                    sample_idx = np.random.choice(flat_teacher_features_explorer.shape[0], sample_num, False)
                    density_samples_explorer = flat_teacher_features_explorer[sample_idx] + cfg_flow['noise'] * torch.randn_like(flat_teacher_features_explorer[sample_idx])
                    z_explorer, log_jacobian_explorer = explorer_density(density_samples_explorer)
                    density_loss_explorer, _, _ = explorer_density_loss_fn(z_explorer, sldj=log_jacobian_explorer)
                    (density_loss_explorer / accumulation_steps).backward()
                    exploiter_density.eval()
                    explorer_density.eval()
                outputs_l_exploiter = exploiter(labeled_ap)
                logits_l_exploiter = outputs_l_exploiter['pred']
                probs_l_exploiter = torch.softmax(logits_l_exploiter, dim=1)
                outputs_l_explorer = explorer(labeled_be)
                logits_l_explorer_unaligned = outputs_l_explorer['pred']
                logits_l_explorer = torch.zeros_like(logits_l_explorer_unaligned)
                for batch_index in range(logits_l_explorer.shape[0]):
                    logits_l_explorer[batch_index] = rotate(logits_l_explorer_unaligned[batch_index], -int(rotate_labeled[batch_index].item()), interpolation=InterpolationMode.NEAREST)
                probs_l_explorer = torch.softmax(logits_l_explorer, dim=1)
                supervised_loss_exploiter = 0.5 * (ce_loss(logits_l_exploiter, labeled_label_batch.long()) + dice_loss(probs_l_exploiter, labeled_label_batch.unsqueeze(1)))
                supervised_loss_explorer = 0.5 * (ce_loss(logits_l_explorer, labeled_label_batch.long()) + dice_loss(probs_l_explorer, labeled_label_batch.unsqueeze(1)))
                supervised_loss_total = supervised_loss_exploiter + supervised_loss_explorer
                (supervised_loss_total / accumulation_steps).backward()
                outputs_u_exploiter = exploiter(unlabeled_ap)
                logits_exploiter, features_exploiter = (outputs_u_exploiter['pred'], outputs_u_exploiter['features'])
                outputs_u_explorer = explorer(unlabeled_be)
                logits_explorer_unaligned, features_explorer = (outputs_u_explorer['pred'], outputs_u_explorer['features'])
                logits_explorer = torch.zeros_like(logits_explorer_unaligned)
                for batch_index in range(logits_explorer.shape[0]):
                    logits_explorer[batch_index] = rotate(logits_explorer_unaligned[batch_index], -int(rotate_unlabeled[batch_index].item()), interpolation=InterpolationMode.NEAREST)
                pseudo_from_exploiter = torch.argmax(torch.softmax(logits_exploiter.detach(), dim=1), dim=1)
                pseudo_from_explorer = torch.argmax(torch.softmax(logits_explorer.detach(), dim=1), dim=1)
                cps_loss_exploiter = ce_loss(logits_exploiter, pseudo_from_explorer)
                cps_loss_explorer = ce_loss(logits_explorer, pseudo_from_exploiter)
                adp_loss_exploiter = torch.tensor(0.0).cuda(local_rank)
                adp_loss_explorer = torch.tensor(0.0).cuda(local_rank)
                if epoch >= adp_start_epoch:
                    features_to_perturb_exploiter = features_exploiter[-1]
                    perturbed_features_exploiter = adp_perturb_feature_asymmetric(features_to_perturb_exploiter, exploiter_density, exploiter_density_loss_fn, cfg_flow['eps'], direction=args.adp_direction_exploiter)
                    aux_features_exploiter = features_exploiter[:-1] + [perturbed_features_exploiter]
                    aux_logits_exploiter = exploiter.module.decoder(aux_features_exploiter)
                    adp_loss_exploiter = ce_loss(aux_logits_exploiter, pseudo_from_explorer)
                    features_to_perturb_explorer = features_explorer[-1]
                    perturbed_features_explorer = adp_perturb_feature_asymmetric(features_to_perturb_explorer, explorer_density, explorer_density_loss_fn, cfg_flow['eps'], direction='descending')
                    aux_features_explorer = features_explorer[:-1] + [perturbed_features_explorer]
                    aux_logits_explorer_unaligned = explorer.module.decoder(aux_features_explorer)
                    aux_logits_explorer = torch.zeros_like(aux_logits_explorer_unaligned)
                    for batch_index in range(aux_logits_explorer.shape[0]):
                        aux_logits_explorer[batch_index] = rotate(aux_logits_explorer_unaligned[batch_index], -int(rotate_unlabeled[batch_index].item()), interpolation=InterpolationMode.NEAREST)
                    adp_loss_explorer = ce_loss(aux_logits_explorer, pseudo_from_exploiter)
                consistency_weight = sigmoid_rampup(iter_num, args.consistency_ramp_iters) * args.consistency_weight_max
                lambda_adp = args.adp_loss_weight
                unsupervised_loss_exploiter = (1.0 - lambda_adp) * cps_loss_exploiter + lambda_adp * adp_loss_exploiter
                unsupervised_loss_explorer = (1.0 - lambda_adp) * cps_loss_explorer + lambda_adp * adp_loss_explorer
                exploiter_loss = supervised_loss_exploiter + consistency_weight * unsupervised_loss_exploiter
                explorer_loss = supervised_loss_explorer + consistency_weight * unsupervised_loss_explorer
                loss = exploiter_loss + explorer_loss
                unsupervised_loss_total = consistency_weight * (unsupervised_loss_exploiter + unsupervised_loss_explorer)
                (unsupervised_loss_total / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    if cfg_flow['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(exploiter.parameters(), cfg_flow['grad_clip'])
                        torch.nn.utils.clip_grad_norm_(explorer.parameters(), cfg_flow['grad_clip'])
                        if epoch >= adp_start_epoch:
                            torch.nn.utils.clip_grad_norm_(exploiter_density.parameters(), cfg_flow['grad_clip'])
                            torch.nn.utils.clip_grad_norm_(explorer_density.parameters(), cfg_flow['grad_clip'])
                    exploiter_optimizer.step()
                    explorer_optimizer.step()
                    if epoch >= adp_start_epoch:
                        exploiter_density_optimizer.step()
                        explorer_density_optimizer.step()
                    exploiter_optimizer.zero_grad()
                    explorer_optimizer.zero_grad()
                    exploiter_density_optimizer.zero_grad()
                    explorer_density_optimizer.zero_grad()
                    iter_num += 1
                    with torch.no_grad():
                        ema_decay = cfg_flow['ema_decay']
                        for t_params, s_params in zip(exploiter_teacher.parameters(), exploiter.parameters()):
                            t_params.data.copy_(ema_decay * t_params.data + (1.0 - ema_decay) * s_params.data)
                        for t_params, s_params in zip(explorer_teacher.parameters(), explorer.parameters()):
                            t_params.data.copy_(ema_decay * t_params.data + (1.0 - ema_decay) * s_params.data)
                    lr_ = cfg['lr'] * (1 - iter_num / max_iterations) ** 0.9
                    for param_group in exploiter_optimizer.param_groups:
                        param_group['lr'] = lr_
                    for param_group in explorer_optimizer.param_groups:
                        param_group['lr'] = lr_
                if (i + 1) % accumulation_steps == 0:
                    values = [loss, supervised_loss_exploiter, supervised_loss_explorer, cps_loss_exploiter, cps_loss_explorer, adp_loss_exploiter, adp_loss_explorer]
                    if not all((torch.isfinite(v).all().item() for v in values)):
                        raise FloatingPointError(f'Nonfinite loss at update {iter_num}')
                    writer.writerow(dict(zip(fields, [iter_num, epoch, *[v.item() for v in values], consistency_weight, epoch >= adp_start_epoch])))
                    log.flush()
                    print(f'update={iter_num}/{max_iterations} loss={loss.item():.6f}', flush=True)
                    if iter_num % cfg['validation_interval'] == 0 or iter_num == max_iterations:
                        cfg['evaluation_iter'] = iter_num
                        exploiter.eval()
                        explorer.eval()
                        a, _, b, _, _ = protocol_joint_validation(exploiter, explorer, valloader, cfg)
                        checkpoint = dict(epoch=epoch, iter_num=iter_num, protocol_config=cfg,
                            exploiter_state_dict=exploiter.module.state_dict(), explorer_state_dict=explorer.module.state_dict(),
                            exploiter_teacher_state_dict=exploiter_teacher.module.state_dict(), explorer_teacher_state_dict=explorer_teacher.module.state_dict(),
                            exploiter_density_state_dict=exploiter_density.module.state_dict(), explorer_density_state_dict=explorer_density.module.state_dict(),
                            exploiter_optimizer_state_dict=exploiter_optimizer.state_dict(), explorer_optimizer_state_dict=explorer_optimizer.state_dict(),
                            exploiter_density_optimizer_state_dict=exploiter_density_optimizer.state_dict(), explorer_density_optimizer_state_dict=explorer_density_optimizer.state_dict())
                        record_snapshot(checkpoint, snapshot_path, a, b, cfg)
                        print(f'validation update={iter_num} exploiter={a:.9f} explorer={b:.9f}', flush=True)
                if iter_num >= max_iterations:
                    break
            if iter_num >= max_iterations:
                break
    marker = 'SMOKE_COMPLETE.json' if cfg['smoke'] else 'TRAIN_COMPLETE.json'
    (Path(snapshot_path) / marker).write_text(json.dumps(dict(iterations=iter_num, seed=cfg['seed'], smoke=cfg['smoke'])))
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
