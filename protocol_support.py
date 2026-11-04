"""Own-branch validation selection plus a synchronized pair for diagnostics."""
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from val_3D import test_single_case_efficient_BB
from util.checkpoints import ROLES
from util.run_contract import choose_best


def audit_training_state(checkpoint):
    models = [role+suffix for role in ROLES
              for suffix in ('_state_dict', '_teacher_state_dict', '_density_state_dict')]
    optimizers = [role+suffix for role in ROLES
                  for suffix in ('_optimizer_state_dict', '_density_optimizer_state_dict')]
    def finite(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or torch.isfinite(value).all().item()
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(x) for x in value)
        return True
    assert all(checkpoint[key] and finite(checkpoint[key]) for key in models)
    assert all(checkpoint[key]['state'] and finite(checkpoint[key]) for key in optimizers)
    return dict(passed=True, finite_models=len(models), finite_optimizers=len(optimizers),
                per_iteration_snapshot_archives=False)


def append_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    fresh = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def protocol_joint_validation(exploiter, explorer, valloader, cfg):
    expected = Path(cfg['val_list']).read_text().splitlines()
    assert valloader.dataset.mode == 'val' and valloader.dataset.ids == expected
    scores = []
    for role, net in (('exploiter', exploiter), ('explorer', explorer)):
        net.eval()
        cases = []
        for batch in valloader:
            pred, _ = test_single_case_efficient_BB(net, batch['image'][0].numpy(),
                cfg['eval_stride_xy'], cfg['eval_stride_z'], cfg['eval_patch'], num_classes=2)
            gt = batch['label'][0].numpy()
            p, g = pred > 0, gt > 0
            score = 2 * np.count_nonzero(p & g) / max(1, np.count_nonzero(p) + np.count_nonzero(g))
            cases.append(dict(iter=cfg['evaluation_iter'], role=role, case=batch['id'][0], dice=float(score)))
            if cfg.get('smoke'):
                break
        used = [expected[int(next(iter(valloader.sampler)))]] if cfg.get('smoke') else expected
        assert len(cases) == len(used) and {c['case'] for c in cases} == set(used)
        append_rows(Path(cfg['output_dir']) / 'validation_case_dice.csv', cases)
        scores.append(float(np.mean([c['dice'] for c in cases])))
    return scores[0], [scores[0]], scores[1], [scores[1]], {}


def record_snapshot(checkpoint, snapshot_path, exploiter_dice, explorer_dice, cfg):
    root = Path(snapshot_path)
    checkpoint['rng_state'] = dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
    checkpoint['rng_resume_limit'] = 'DataLoader worker/prefetch states are not saved; no bitwise resume claim.'
    torch.save(checkpoint, root / 'last.pth')
    own_path = root / 'branch_selection.json'
    own = json.loads(own_path.read_text()) if own_path.exists() else {}
    for role, score in [('exploiter', exploiter_dice), ('explorer', explorer_dice)]:
        if role not in own or float(score) > own[role]['validation_dice']:
            filename = 'best_'+role+'.pth'
            torch.save(checkpoint[role+'_state_dict'], root / filename)
            own[role] = dict(iter=int(checkpoint['iter_num']), validation_dice=float(score),
                             weight_file=filename, architecture=cfg['role_architectures'][role])
    own_path.write_text(json.dumps(own, indent=2))
    mean = (float(exploiter_dice) + float(explorer_dice)) / 2
    selection = root / 'selection.json'
    previous = json.loads(selection.read_text()) if selection.exists() else {'mean_validation_dice': -1}
    if mean > previous['mean_validation_dice']:
        torch.save(checkpoint, root / 'pair_best.pth')
        for role in ('exploiter', 'explorer'):
            torch.save(checkpoint[f'{role}_state_dict'], root / f'pair_best_{role}.pth')
        selection.write_text(json.dumps(dict(iter=checkpoint['iter_num'], mean_validation_dice=mean,
            exploiter_validation_dice=float(exploiter_dice), explorer_validation_dice=float(explorer_dice),
            rule='smoke_not_a_scientific_score' if cfg.get('smoke') else 'highest_mean_two_branch_validation_dice; earliest_tie'), indent=2))
    append_rows(root / 'validation_pair_history.csv', [dict(iter=checkpoint['iter_num'],
        exploiter_dice=float(exploiter_dice), explorer_dice=float(explorer_dice), mean_dice=mean)])
