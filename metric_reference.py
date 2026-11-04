"""Two CPU implementations with the retained empty-mask conventions."""


def metrics(pred, gt, cls, independent=False):
    import numpy as np
    from mechanism_metrics import class_metrics
    from scipy import ndimage as ndi
    p, g = pred == cls, gt == cls
    n, m, intersection = int(p.sum()), int(g.sum()), int((p & g).sum())
    if independent:
        result = {'class': cls, 'tp': intersection, 'fp': n-intersection, 'fn': m-intersection,
                  'dice': 2*intersection/(n+m) if n+m else 1.,
                  'jaccard': intersection/(n+m-intersection) if n+m-intersection else 1.,
                  'precision': intersection/n if n else float('nan'),
                  'recall': intersection/m if m else float('nan'),
                  'hd95_vox': float('nan'), 'asd_pred_to_gt_vox': float('nan'),
                  'assd_vox': float('nan'), 'surface_status': 'empty_mask'}
        if n and m:
            structure = ndi.generate_binary_structure(3, 1)
            ps = p ^ ndi.binary_erosion(p, structure, border_value=0)
            gs = g ^ ndi.binary_erosion(g, structure, border_value=0)
            a = ndi.distance_transform_edt(~gs)[ps]
            b = ndi.distance_transform_edt(~ps)[gs]
            result.update(hd95_vox=float(np.percentile(np.concatenate((a, b)), 95)),
                          asd_pred_to_gt_vox=float(a.mean()), assd_vox=float((a.mean()+b.mean())/2),
                          surface_status='ok')
    else:
        result = class_metrics(pred, gt, cls)
        if not n and not m:
            result.update(dice=1., jaccard=1.)
    result.update(pred_voxels=n, gt_voxels=m, distance_unit='voxel',
                  empty_status='both_empty' if not n and not m else 'one_empty' if not n or not m else 'nonempty')
    return result


def same(a, b):
    import numpy as np
    assert a.keys() == b.keys()
    for key, value in a.items():
        if isinstance(value, float):
            assert np.isclose(value, b[key], rtol=1e-8, atol=1e-9, equal_nan=True), (key, value, b[key])
        else:
            assert value == b[key], (key, value, b[key])


def fixtures():
    import numpy as np
    a = np.zeros((8, 9, 10), np.uint8)
    a[1:4, 1:4, 1:4] = 1
    a[4:6, 2:5, 3:7] = 2
    a[2:4, 6:8, 6:9] = 3
    b = np.roll(a, 1, axis=0)
    checks = 0
    for pred, gt in [(a, a), (a, b), (a, np.zeros_like(a)), (np.zeros_like(a), a),
                     (np.zeros_like(a), np.zeros_like(a))]:
        for cls in range(4):
            same(metrics(pred, gt, cls), metrics(pred, gt, cls, independent=True))
            checks += 1
    assert [metrics(a, a, c)['dice'] for c in range(4)] == [1.] * 4
    return dict(passed=True, independent_metric_checks=checks, all_four_classes=True,
                both_empty_overlap_one=True, undefined_surface_NaN=True, shifted_and_one_empty=True)
