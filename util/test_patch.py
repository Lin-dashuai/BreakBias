import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from skimage.measure import label

def GetLargest_CC(segmentation):
    """
    获取一个分割图中最大的连通区域。

    Args:
        segmentation (np.array): 模型的预测分割图。

    Returns:
        np.array: 只包含最大连通区域的分割图。
    """
    segmentation_int = segmentation.astype(np.int32)
    binary_segmentation = segmentation_int > 0
    labels = label(binary_segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    final_segmentation = np.zeros_like(segmentation, dtype=segmentation.dtype)
    final_segmentation[largestCC] = segmentation[largestCC]
    return final_segmentation

def test_single_case_efficient_BB(model, image, stride_xy, stride_z, patch_size, num_classes=4, batch_size=16, save_path=None, affine=np.eye(4)):
    image_fp32 = image.astype(np.float32)
    image_tensor = torch.from_numpy(image_fp32).cuda(non_blocking=True)
    w, h, d = image_tensor.shape
    add_pad = False
    pad_dims = [0, 0, 0, 0, 0, 0]
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
        pad_dims[5], pad_dims[4] = (w_pad // 2, w_pad - w_pad // 2)
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
        pad_dims[3], pad_dims[2] = (h_pad // 2, h_pad - h_pad // 2)
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
        pad_dims[1], pad_dims[0] = (d_pad // 2, d_pad - d_pad // 2)
    if add_pad:
        image_tensor = F.pad(image_tensor, pad_dims, mode='constant', value=0)
    ww, hh, dd = image_tensor.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = torch.zeros((num_classes,) + (ww, hh, dd), device='cuda')
    cnt = torch.zeros((ww, hh, dd), device='cuda')

    def coord_generator():
        for x in range(sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(sz):
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
            patch_batch = torch.stack([image_tensor[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] for xs, ys, zs in coords_batch]).unsqueeze(1)
            out_dict = model(patch_batch)
            y = out_dict['pred']
            y = F.softmax(y, dim=1)
            for i, (xs, ys, zs) in enumerate(coords_batch):
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y[i]
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
            if len(coords_batch) < batch_size:
                break
    score_map = score_map / (cnt.unsqueeze(0) + 1e-06)
    label_map = torch.argmax(score_map, dim=0)
    if add_pad:
        label_map = label_map[pad_dims[4]:ww - pad_dims[5], pad_dims[2]:hh - pad_dims[3], pad_dims[0]:dd - pad_dims[1]]
    label_map_np = label_map.cpu().numpy().astype(np.uint8)
    if save_path is not None:
        try:
            nifti_img = nib.Nifti1Image(label_map_np, affine)
            nib.save(nifti_img, save_path)
        except Exception as e:
            print(f'  -> ERROR: Failed to save Nifti file to {save_path}. Reason: {e}')
    return (label_map_np, None)
