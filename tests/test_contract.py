import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from util.checkpoints import ARCHITECTURES, normalize_config
from util.run_contract import PUBLISHED_WEIGHTS, choose_best, partition_contract, verify_run, verify_weight_bundle

ROOT = Path(__file__).resolve().parents[1]


class ContractFixtures(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'configs/la5_f02.json').read_text())

    def test_f02_recipe_only(self):
        cfg = normalize_config(self.config())
        self.assertEqual(cfg['role_architectures'], ARCHITECTURES)
        self.assertEqual(cfg['adp']['eps'], .0075)
        cfg['recipe'] = 'unsupported'
        with self.assertRaises(ValueError):
            normalize_config(cfg)

    def test_bad_scientific_settings_rejected(self):
        for key, value in [('crop_size', [96, 96, 96]), ('max_iterations', 1500),
                           ('consistency_ramp_iters', 2000), ('role_architectures', {'explorer': 'V', 'exploiter': 'U'})]:
            cfg = self.config(); cfg[key] = value
            with self.assertRaises(ValueError):
                normalize_config(cfg)
        for key, value in [('eps', .01), ('eps', .005), ('loss_weight', .7)]:
            cfg = self.config(); cfg['adp'][key] = value
            with self.assertRaises(ValueError):
                normalize_config(cfg)

    def test_seed_and_copy_contract(self):
        cfg = self.config(); before = copy.deepcopy(cfg)
        changed = normalize_config(cfg); changed['adp']['eps'] = 9
        self.assertEqual(cfg, before)
        for seed in [True, -1, 1.5]:
            cfg = self.config(); cfg['seed'] = seed
            with self.assertRaises(ValueError):
                normalize_config(cfg)

    def test_smoke_is_not_formal(self):
        cfg = self.config(); cfg.update(smoke=True, max_iterations=2); cfg['adp']['start_epoch'] = 0
        with self.assertRaises(ValueError):
            normalize_config(cfg)
        self.assertTrue(normalize_config(cfg, allow_smoke=True)['smoke'])

    def test_partitions_disjoint(self):
        value = partition_contract(ROOT)
        self.assertEqual((value['train'], value['validation'], value['test']), (80, 10, 10))
        self.assertTrue(value['all_disjoint'])

    def test_own_best_earliest_tie(self):
        rows = [dict(iter=250, explorer_dice=.8, exploiter_dice=.5),
                dict(iter=500, explorer_dice=.6, exploiter_dice=.9),
                dict(iter=750, explorer_dice=.8, exploiter_dice=.9)]
        self.assertEqual(choose_best(rows, 'explorer')['iter'], 250)
        self.assertEqual(choose_best(rows, 'exploiter')['iter'], 500)

    def fake_run(self):
        run = Path(tempfile.mkdtemp(prefix='breakbias_contract_'))
        def cleanup():
            for path in run.iterdir():
                path.unlink()
            run.rmdir()
        self.addCleanup(cleanup)
        def write(name, value):
            (run/name).write_text(json.dumps(value))
        cfg = self.config(); cfg['smoke'] = False
        write('effective_config.json', cfg)
        write('TRAIN_COMPLETE.json', dict(iterations=6000, state_audit_passed=True, smoke=False,
                                          recipe=cfg['recipe'], seed=cfg['seed']))
        write('training_audit.json', dict(passed=True, finite_models=6, finite_optimizers=4))
        history = []; cases = []; ids = partition_contract(ROOT)['validation_ids']
        for iteration in range(250, 6001, 250):
            u = .8 if iteration in [750, 1000] else .6
            v = .85 if iteration in [500, 1000] else .5
            history.append(dict(iter=iteration, explorer_dice=u, exploiter_dice=v, mean_dice=(u+v)/2))
            for role, score in [('explorer', u), ('exploiter', v)]:
                cases.extend(dict(iter=iteration, role=role, case=case, dice=score) for case in ids)
        for name, rows in [('validation_pair_history.csv', history), ('validation_case_dice.csv', cases)]:
            with (run/name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        selected = {}
        for role in ['explorer', 'exploiter']:
            row = choose_best(history, role)
            selected[role] = dict(iter=row['iter'], validation_dice=row[role+'_dice'],
                                  architecture=ARCHITECTURES[role], weight_file='best_'+role+'.pth')
            (run/selected[role]['weight_file']).touch()
        write('branch_selection.json', selected)
        return run, selected

    def test_full_run_contract_different_branch_iterations(self):
        run, selected = self.fake_run()
        self.assertEqual(selected['explorer']['iter'], 750)
        self.assertEqual(selected['exploiter']['iter'], 500)
        verify_run(run, ROOT)

    def test_pair_best_cannot_replace_own_selection(self):
        run, selected = self.fake_run(); selected['explorer']['iter'] = 1000
        (run/'branch_selection.json').write_text(json.dumps(selected))
        with self.assertRaises(ValueError):
            verify_run(run, ROOT)

    def test_missing_final_validation_rejected(self):
        run, _ = self.fake_run()
        path = run/'validation_pair_history.csv'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
        with self.assertRaises(ValueError):
            verify_run(run, ROOT)

    def test_published_weight_bundle(self):
        run = ROOT/'weights/la5_f02_v10_seed42'
        cfg, selected, _ = verify_weight_bundle(run, ROOT)
        self.assertEqual(cfg['role_architectures'], ARCHITECTURES)
        self.assertEqual({path.name for path in run.iterdir()},
                         {'best_explorer.pth', 'best_exploiter.pth'})
        expected_paths = set()
        for role, entry in PUBLISHED_WEIGHTS.items():
            path = run/entry['file']
            expected_paths.add(path)
            raw = path.read_bytes()
            self.assertEqual(len(raw), entry['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['sha256'])
            self.assertEqual(entry['file'], selected[role]['weight_file'])
        self.assertEqual(set(ROOT.rglob('*.pth')), expected_paths)

    def fake_weight_bundle(self):
        run = Path(tempfile.mkdtemp(prefix='breakbias_weights_'))
        def cleanup():
            for path in run.iterdir():
                path.unlink()
            run.rmdir()
        self.addCleanup(cleanup)
        weights = copy.deepcopy(PUBLISHED_WEIGHTS)
        for role, entry in weights.items():
            raw = role.encode('ascii')
            (run/entry['file']).write_bytes(raw)
            entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        override = patch('util.run_contract.PUBLISHED_WEIGHTS', weights)
        override.start()
        self.addCleanup(override.stop)
        return run, weights

    def test_weight_bundle_without_training_records(self):
        run, _ = self.fake_weight_bundle()
        cfg, _, _ = verify_weight_bundle(run, ROOT)
        self.assertNotIn('seed', cfg)
        self.assertFalse((run/'validation_pair_history.csv').exists())

    def test_weight_bundle_checksum_rejected(self):
        run, _ = self.fake_weight_bundle()
        (run/'best_explorer.pth').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            verify_weight_bundle(run, ROOT)

    def test_weight_bundle_path_rejected(self):
        run, weights = self.fake_weight_bundle()
        weights['explorer']['file'] = '../best_explorer.pth'
        with self.assertRaises(ValueError):
            verify_weight_bundle(run, ROOT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
