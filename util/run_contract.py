"""Validate patient partitions and own-branch selection without loading weights."""
import csv
import hashlib
import json
import math
from pathlib import Path
from util.checkpoints import ARCHITECTURES, ROLES, normalize_config


PUBLISHED_WEIGHTS = {
    'exploiter': dict(file='best_exploiter.pth', architecture='V', n_filters=16,
                      bytes=37892074, sha256='1aa09750fde4de0fd1eb088460692ac1096ff36b22bffec3181aaa86a65b3828'),
    'explorer': dict(file='best_explorer.pth', architecture='U', n_filters=16,
                     bytes=13000079, sha256='0716021a9a4ab91a54c3596cb03efbe7709f1aba3c4d72881425e32e892234ba'),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def partition_contract(root):
    root = Path(root)/'partitions/LA'
    def ids(path):
        values = path.read_text().splitlines()
        if not values or len(values) != len(set(values)):
            raise ValueError('Empty or duplicate patient list: '+str(path))
        if any('/' in value or '\\' in value or value in ('.', '..') for value in values):
            raise ValueError('Unsafe patient identifier')
        return values
    val, test = ids(root/'val.list'), ids(root/'test.list')
    if len(val) != 10 or len(test) != 10 or set(val) & set(test):
        raise ValueError('Expected disjoint val10/test10')
    train = None
    for denominator, count in [(20, 4), (10, 8), (5, 16)]:
        labeled = ids(root/f'1_{denominator}/labeled.list')
        unlabeled = ids(root/f'1_{denominator}/unlabeled.list')
        cohort = set(labeled+unlabeled)
        if len(labeled) != count or len(unlabeled) != 80-count or len(cohort) != 80:
            raise ValueError('Invalid labeled/unlabeled partition')
        if cohort & set(val+test) or (train is not None and cohort != train):
            raise ValueError('Patient partitions overlap or change across label ratios')
        train = cohort
    return dict(train=80, validation=10, test=10, all_disjoint=True, validation_ids=val, test_ids=test)


def choose_best(history, role):
    if role not in ROLES or not history:
        raise ValueError('Missing role or validation history')
    return min(history, key=lambda row: (-float(row[role+'_dice']), int(row['iter'])))


def verify_weight_bundle(run, root):
    run = Path(run)
    cfg = dict(image_channel=1, nclass=2, eval_patch=[112, 112, 80],
               eval_stride_xy=18, eval_stride_z=4)
    if set(PUBLISHED_WEIGHTS) != set(ROLES):
        raise ValueError('Expected one weight file for each role')
    cfg['role_architectures'] = dict(ARCHITECTURES)
    selected = {}
    for role in ROLES:
        entry = PUBLISHED_WEIGHTS[role]
        name = 'best_'+role+'.pth'
        if entry['file'] != name or entry['architecture'] != ARCHITECTURES[role] or entry['n_filters'] != 16:
            raise ValueError('Role/weight binding mismatch')
        weight = run/name
        if not weight.is_file() or weight.is_symlink():
            raise ValueError('Missing or unsafe weight file')
        raw = weight.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Weight checksum mismatch')
        cfg[role+'_filters'] = entry['n_filters']
        selected[role] = dict(weight_file=name, architecture=entry['architecture'])
    return cfg, selected, partition_contract(root)


def verify_run(run, root, smoke=False):
    run = Path(run)
    cfg = normalize_config(read_json(run/'effective_config.json'), allow_smoke=smoke)
    marker = read_json(run/('SMOKE_COMPLETE.json' if smoke else 'TRAIN_COMPLETE.json'))
    if bool(cfg.get('smoke')) != smoke or marker['smoke'] != smoke:
        raise ValueError('Smoke and formal outputs cannot be mixed')
    if marker['iterations'] != (2 if smoke else 6000) or not marker['state_audit_passed']:
        raise ValueError('Training is incomplete or its state audit failed')
    if marker['recipe'] != cfg['recipe'] or marker['seed'] != cfg['seed']:
        raise ValueError('Completion marker does not match the effective configuration')
    audit = read_json(run/'training_audit.json')
    if not audit['passed'] or audit['finite_models'] != 6 or audit['finite_optimizers'] != 4:
        raise ValueError('Expected six finite models and four finite optimizer states')
    partitions = partition_contract(root)
    history = read_csv(run/'validation_pair_history.csv')
    expected_iterations = [2] if smoke else list(range(250, 6001, 250))
    if [int(row['iter']) for row in history] != expected_iterations:
        raise ValueError('A formal run requires all 24 complete validation points')
    cases = read_csv(run/'validation_case_dice.csv')
    for row in history:
        for role in ROLES:
            own = [x for x in cases if int(x['iter']) == int(row['iter']) and x['role'] == role]
            expected_count = 1 if smoke else 10
            if len(own) != expected_count or len({x['case'] for x in own}) != expected_count:
                raise ValueError('Incomplete per-patient validation')
            if not {x['case'] for x in own} <= set(partitions['validation_ids']):
                raise ValueError('Validation cases do not belong to val.list')
            if not math.isclose(math.fsum(float(x['dice']) for x in own)/expected_count,
                                float(row[role+'_dice']), abs_tol=1e-12):
                raise ValueError('Validation mean does not match per-patient scores')
    selected = read_json(run/'branch_selection.json')
    for role in ROLES:
        best = choose_best(history, role)
        actual = selected[role]
        if actual['iter'] != int(best['iter']) or not math.isclose(
                actual['validation_dice'], float(best[role+'_dice']), abs_tol=1e-12):
            raise ValueError('Branch selection is not its own earliest validation best')
        if actual['architecture'] != ARCHITECTURES[role] or actual['weight_file'] != 'best_'+role+'.pth':
            raise ValueError('Role/weight binding mismatch')
        weight = run/actual['weight_file']
        if not weight.is_file() or weight.is_symlink():
            raise ValueError('Missing or unsafe branch-best weight file')
    return cfg, selected, cases, partitions
