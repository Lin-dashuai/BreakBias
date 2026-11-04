import numpy as np

def ratio(a, b):
    return float(a / b) if b else float('nan')

def class_metrics(pred, gt, cls):
    from medpy.metric.binary import hd95, asd
    p, g = (pred == cls, gt == cls)
    tp, fp, fn = (int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()))
    out = {'class': cls, 'tp': tp, 'fp': fp, 'fn': fn, 'dice': ratio(2 * tp, 2 * tp + fp + fn), 'jaccard': ratio(tp, tp + fp + fn), 'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn), 'hd95_vox': float('nan'), 'asd_pred_to_gt_vox': float('nan'), 'assd_vox': float('nan'), 'surface_status': 'empty_mask'}
    if p.any() and g.any():
        out.update(hd95_vox=float(hd95(p, g)), asd_pred_to_gt_vox=float(asd(p, g)), assd_vox=float((asd(p, g) + asd(g, p)) / 2), surface_status='ok')
    return out
