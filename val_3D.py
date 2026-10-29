import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def _extract_logits(output):
    if isinstance(output, dict):
        return output['pred']
    if isinstance(output, (list, tuple)):
        return output[0]
    return output

def _probabilities_from_logits(logits):
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        return torch.cat([1.0 - p, p], dim=1)
    return F.softmax(logits, dim=1)

def _match_score_channels(prob, num_classes):
    if prob.shape[1] == num_classes:
        return prob
    if num_classes == 1 and prob.shape[1] == 2:
        return prob[:, 1:2]
    raise ValueError(f'Probability channels {prob.shape[1]} do not match num_classes={num_classes}')

def test_single_case_efficient_BB(model, image, stride_xy, stride_z, patch_size, num_classes=4, batch_size=8, return_prob=False):
    image_tensor = torch.from_numpy(image).cuda()
    w, h, d = image_tensor.shape
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    if add_pad:
        wl_pad, wr_pad = (w_pad // 2, w_pad - w_pad // 2)
        hl_pad, hr_pad = (h_pad // 2, h_pad - h_pad // 2)
        dl_pad, dr_pad = (d_pad // 2, d_pad - d_pad // 2)
        image_tensor = F.pad(image_tensor, (dl_pad, dr_pad, hl_pad, hr_pad, wl_pad, wr_pad), mode='constant', value=0)
    ww, hh, dd = image_tensor.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = torch.zeros((num_classes,) + (ww, hh, dd), device='cuda')
    cnt = torch.zeros((ww, hh, dd), device='cuda')

    def coord_generator():
        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    yield (xs, ys, zs)
    gen = coord_generator()
    with torch.no_grad(), autocast():
        while True:
            coords_batch = []
            try:
                for _ in range(batch_size):
                    coords_batch.append(next(gen))
            except StopIteration:
                if not coords_batch:
                    break
            patch_batch = torch.stack([image_tensor[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] for xs, ys, zs in coords_batch])
            patch_batch = patch_batch.unsqueeze(1)
            y = _match_score_channels(_probabilities_from_logits(_extract_logits(model(patch_batch))), num_classes)
            for i, (xs, ys, zs) in enumerate(coords_batch):
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y[i]
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
            if len(coords_batch) < batch_size:
                break
    score_map = score_map / cnt.unsqueeze(0)
    label_map = torch.argmax(score_map, dim=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        if return_prob:
            score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    label_map_np = label_map.cpu().numpy().astype(np.uint8)
    if return_prob:
        return (label_map_np, score_map.cpu().numpy().astype(np.float32))
    return (label_map_np, None)
