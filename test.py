"""Test a plain LA state_dict and save raw/LCC predictions and all class metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from model.UNET3D import UNet3D
from model.VNet import VNetv2
from mechanism_metrics import class_metrics
from protocol_support import append_rows
from util.test_patch import GetLargest_CC


def main():
    parser = argparse.ArgumentParser(description='BreakBias LA V10 checkpoint test')
    parser.add_argument('--preset', choices=['h03-full', 'legacy-full'], default='h03-full')
    parser.add_argument('--checkpoint')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--case-list', default=str(Path(__file__).resolve().parent / 'partitions/LA/test.list'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.preset == 'h03-full':
        from val_3D import test_single_case_efficient_BB
        cls, patch, xy, z = VNetv2, [112, 112, 80], 18, 4
        default = root / 'weights/la5_h03_full_model2_iter2000.pth'
    else:
        from util.test_patch import test_single_case_efficient_BB
        cls, patch, xy, z = UNet3D, [96, 96, 96], 16, 16
        default = root / 'weights/la5_legacy_full_best_model1.pth'
    checkpoint = Path(args.checkpoint) if args.checkpoint else default
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values())
    assert all(not v.is_floating_point() or torch.isfinite(v).all().item() for v in state.values())
    model = cls(n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=True)
    model.load_state_dict(state, strict=True)
    del state
    model.cuda().eval()
    cases = Path(args.case_list).read_text().splitlines()
    assert cases and len(cases) == len(set(cases))
    if args.max_cases is not None:
        assert args.max_cases > 0
        cases = cases[:args.max_cases]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(preset=args.preset, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        patch=patch, stride_xy=xy, stride_z=z, batch_size=8, inference_precision='legacy autocast',
        distance_unit='voxel', postprocessing=['raw', 'legacy_lcc'], cases=cases)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for case in cases:
        assert Path(case).name == case and case not in ('.', '..')
        with h5py.File(Path(args.data_root) / case / 'mri_norm2.h5', 'r') as data:
            image, gt = data['image'][:].astype('float32'), data['label'][:].astype('uint8')
        pred, _ = test_single_case_efficient_BB(model, image, xy, z, patch, num_classes=2, batch_size=8)
        lcc = GetLargest_CC(pred).astype('uint8')
        with h5py.File(out / f'{case}.h5', 'w') as saved:
            saved.create_dataset('prediction_raw', data=pred.astype('uint8'), compression='gzip')
            saved.create_dataset('prediction_lcc', data=lcc, compression='gzip')
        new = [dict(case=case, view=view, **class_metrics(value, gt, cls))
               for view, value in [('raw', pred), ('legacy_lcc', lcc)] for cls in (0, 1)]
        append_rows(out / 'per_case_class.csv', new)
        rows.extend(new)
        print(f'completed={len(rows)//4}/{len(cases)} case={case}', flush=True)
    summary = []
    for view in ('raw', 'legacy_lcc'):
        for cls in (0, 1):
            selected = [r for r in rows if r['view'] == view and r['class'] == cls]
            summary.append(dict(view=view, **{'class': cls}, n_cases=len(selected),
                n_valid_surface=sum(r['surface_status'] == 'ok' for r in selected),
                **{key: float(np.nanmean([r[key] for r in selected])) for key in
                   ('dice', 'jaccard', 'precision', 'recall', 'hd95_vox', 'asd_pred_to_gt_vox', 'assd_vox')}))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'TEST_COMPLETE.json').write_text(json.dumps(dict(cases=len(cases), prediction_files=len(cases))))


if __name__ == '__main__':
    main()
