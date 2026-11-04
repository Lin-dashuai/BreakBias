"""Evaluate both branches from a training run or a weight bundle."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import h5py
import numpy as np
import torch
from model.UNET3D import UNet3D
from model.VNet import VNetv2
from metric_reference import metrics, same, fixtures
from util.checkpoints import ROLES, model_state
from util.run_contract import verify_run, verify_weight_bundle
from util.test_patch import GetLargest_CC
from val_3D import test_single_case_efficient_BB

KEYS = ['dice', 'jaccard', 'precision', 'recall', 'hd95_vox', 'asd_pred_to_gt_vox', 'assd_vox']


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    signature = lambda s: [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink]
    assert signature(before) == signature(after)
    return dict(path=str(path), sha256=digest, signature=signature(after))


def summarize(rows):
    output = []
    for role in ROLES:
        for mode in ['raw', 'legacy_lcc']:
            for cls in [0, 1]:
                own = [r for r in rows if r['role'] == role and r['mode'] == mode and r['class'] == cls]
                summary = dict(role=role, mode=mode, **{'class': cls}, cases=len(own), valid_counts={})
                for key in KEYS:
                    values = [float(r[key]) for r in own if math.isfinite(float(r[key]))]
                    summary['valid_counts'][key] = len(values)
                    summary[key] = math.fsum(values)/len(values) if values else None
                output.append(summary)
    return output


def save_prediction(directory, case, predictions, gt):
    payload = dict(gt=gt)
    for role, (pred, prob) in predictions.items():
        payload[role+'_raw'] = pred.astype('uint8')
        payload[role+'_lcc'] = GetLargest_CC(pred).astype('uint8')
        payload[role+'_probability'] = prob
    path = Path(directory)/(case+'.npz')
    if path.exists():
        raise ValueError('Refusing prediction overwrite')
    np.savez_compressed(path, **payload)
    rows = []
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved['gt'], gt)
        for role, (pred, prob) in predictions.items():
            assert np.isfinite(prob).all() and np.allclose(prob.sum(0), 1, atol=.002, rtol=0)
            assert np.array_equal(prob.argmax(0), pred)
            assert np.array_equal(saved[role+'_probability'], prob)
            for mode, suffix in [('raw', '_raw'), ('legacy_lcc', '_lcc')]:
                mask = pred if mode == 'raw' else GetLargest_CC(pred)
                assert np.array_equal(saved[role+suffix], mask)
                for cls in [0, 1]:
                    first = metrics(mask, gt, cls)
                    second = metrics(saved[role+suffix], saved['gt'], cls, independent=True)
                    same(first, second)
                    rows.append(dict(case=case, role=role, mode=mode, **first))
    return rows, fingerprint(path)


def main():
    parser = argparse.ArgumentParser(description='Test both BreakBias branches')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true', help='Validation-only evaluation of a smoke run.')
    parser.add_argument('--weights-only', action='store_true', help='Load published weights.')
    args = parser.parse_args()
    if args.weights_only and args.smoke:
        parser.error('--weights-only and --smoke cannot be combined')
    root = Path(__file__).resolve().parent
    run = Path(args.run_dir).resolve()
    if args.weights_only:
        cfg, selection, partitions = verify_weight_bundle(run, root)
        validation_rows = []
    else:
        cfg, selection, validation_rows, partitions = verify_run(run, root, smoke=args.smoke)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions_dir = output/'predictions'
    predictions_dir.mkdir()
    source_paths = [root/'test.py', root/'val_3D.py', root/'util/run_contract.py',
                    root/'util/checkpoints.py', root/'partitions/LA/val.list', root/'partitions/LA/test.list']
    if not args.weights_only:
        source_paths.extend([run/'effective_config.json', run/'branch_selection.json'])
    sources = {str(p): fingerprint(p) for p in source_paths}
    fixture_result = fixtures()
    models = {}; weights = {}
    for role in ROLES:
        weight = run/selection[role]['weight_file']
        weights[role] = fingerprint(weight)
        cls = UNet3D if cfg['role_architectures'][role] == 'U' else VNetv2
        model = cls(n_channels=1, n_classes=2, n_filters=cfg[role+'_filters'],
                    normalization='batchnorm', has_dropout=True)
        model.load_state_dict(model_state(torch.load(weight, map_location='cpu', weights_only=True)), strict=True)
        models[role] = model.cuda().eval()
    data_signatures = {}
    def case_data(case):
        path = Path(args.data_root)/case/'mri_norm2.h5'
        s = path.stat()
        signature = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink]
        if str(path) in data_signatures:
            assert data_signatures[str(path)] == signature
        data_signatures[str(path)] = signature
        with h5py.File(path, 'r') as stream:
            return stream['image'][:], stream['label'][:]
    def infer(role, image):
        return test_single_case_efficient_BB(models[role], image, cfg['eval_stride_xy'],
            cfg['eval_stride_z'], cfg['eval_patch'], num_classes=2, batch_size=8)
    reproductions = []
    for role in ROLES:
        if args.weights_only:
            continue
        expected = [r for r in validation_rows if r['role'] == role and int(r['iter']) == selection[role]['iter']]
        assert len(expected) == (1 if args.smoke else 10)
        for row in expected:
            image, gt = case_data(row['case'])
            pred, _ = infer(role, image)
            p, g = pred > 0, gt > 0
            score = 2*np.count_nonzero(p & g)/max(1, np.count_nonzero(p)+np.count_nonzero(g))
            assert math.isclose(score, float(row['dice']), abs_tol=1e-5), (role, row['case'], score, row['dice'])
            reproductions.append(dict(role=role, case=row['case'], iteration=selection[role]['iter'],
                                       recorded=float(row['dice']), reproduced=float(score)))
    if not args.weights_only:
        (output/'validation_reproduction.json').write_text(json.dumps(dict(passed=True, rows=reproductions), indent=2))
    cases = [reproductions[0]['case']] if args.smoke else partitions['test_ids']
    rows = []; inventory = []
    for case in cases:
        image, gt = case_data(case)
        predictions = {role: infer(role, image) for role in ROLES}
        own, entry = save_prediction(predictions_dir, case, predictions, gt)
        rows.extend(own); inventory.append(entry)
    with (output/'per_case_class_raw_lcc.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for role in ROLES:
        assert fingerprint(weights[role]['path']) == weights[role]
    for path, expected in sources.items():
        assert fingerprint(path) == expected
    for path, expected in data_signatures.items():
        s = Path(path).stat()
        assert [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink] == expected
    audit = dict(passed=True, primary_role='explorer', primary_mode='raw',
                 selection=selection, selected_weights=weights, prediction_inventory=inventory,
                 summary=summarize(rows), class_rows=len(rows), validation_reproductions=len(reproductions),
                 native_metric_fixtures=fixture_result, saved_predictions_dual_CPU_verified=True,
                 source_data_weights_unchanged=True, pair_best_loaded=False, recovery_loaded=False,
                 protected_sources=sources, data_signatures=data_signatures,
                 smoke=args.smoke, weights_only=args.weights_only, test_cases=0 if args.smoke else len(cases))
    if not args.weights_only:
        audit.update(recipe=cfg['recipe'], seed=cfg['seed'])
    (output/'final_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    marker = 'SMOKE_EVAL_COMPLETE.json' if args.smoke else (
        'TEST_COMPLETE.json' if args.weights_only else 'INDIVIDUAL_BEST_TEST_DONE.json')
    (output/marker).write_text(json.dumps(dict(passed=True, audit=fingerprint(output/'final_audit.json')), indent=2))
    print(json.dumps(dict(passed=True, output=str(output), test_cases=audit['test_cases'], class_rows=len(rows))))


if __name__ == '__main__':
    main()
