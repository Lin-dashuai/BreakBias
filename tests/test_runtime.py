import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import numpy as np
    from metric_reference import fixtures
    from protocol_support import record_snapshot, audit_training_state
    from util.checkpoints import model_state
    import test as evaluation
    RUNTIME = True
except ImportError:
    RUNTIME = False


@unittest.skipUnless(RUNTIME, 'Install requirements.txt to run tensor and metric fixtures')
class RuntimeFixtures(unittest.TestCase):
    def directory(self):
        root = Path(tempfile.mkdtemp(prefix='breakbias_runtime_'))
        def cleanup():
            for path in root.iterdir():
                path.unlink()
            root.rmdir()
        self.addCleanup(cleanup)
        return root

    def test_independent_metrics(self):
        self.assertEqual(fixtures()['independent_metric_checks'], 20)

    def test_branch_best_archives_and_earliest_tie(self):
        root = self.directory()
        cfg = dict(role_architectures={'exploiter': 'V', 'explorer': 'U'}, smoke=False)
        def checkpoint(iteration):
            result = dict(iter_num=iteration)
            for role in ['exploiter', 'explorer']:
                for suffix in ['_state_dict', '_teacher_state_dict', '_density_state_dict']:
                    result[role+suffix] = {'weight': torch.tensor([float(iteration)])}
                for suffix in ['_optimizer_state_dict', '_density_optimizer_state_dict']:
                    result[role+suffix] = {'state': {0: {'momentum': torch.tensor([1.])}}}
            return result
        record_snapshot(checkpoint(250), root, .8, .7, cfg)
        record_snapshot(checkpoint(500), root, .8, .75, cfg)
        selected = json.loads((root/'branch_selection.json').read_text())
        self.assertEqual(selected['exploiter']['iter'], 250)
        self.assertEqual(selected['explorer']['iter'], 500)
        self.assertEqual(audit_training_state(checkpoint(500))['finite_models'], 6)
        self.assertEqual(len(list(root.glob('*.pth'))), 6)
        self.assertFalse((root/'temporal_checkpoints').exists())

    def test_recovery_or_nonfinite_not_plain_inference_weights(self):
        with self.assertRaises(ValueError):
            model_state({'explorer_state_dict': {'weight': torch.tensor([1.])}})
        with self.assertRaises(ValueError):
            model_state({'weight': torch.tensor([float('nan')])})
        self.assertTrue(model_state({'weight': torch.tensor([1.])}))

    def test_prediction_readback_and_nan_counts(self):
        root = self.directory()
        gt = np.zeros((8, 9, 10), np.uint8)
        gt[2:5, 2:5, 2:5] = 1
        pred = gt.copy()
        prob = np.stack([1-pred, pred]).astype('float32')
        rows, _ = evaluation.save_prediction(root, 'fixture',
            {'exploiter': (pred, prob), 'explorer': (pred, prob)}, gt)
        self.assertEqual(len(rows), 8)
        summary = evaluation.summarize(rows)
        self.assertEqual(len(summary), 8)
        self.assertTrue(all(row['dice'] == 1 for row in summary))
        empty = np.zeros_like(gt)
        empty_prob = np.stack([1-empty, empty]).astype('float32')
        rows, _ = evaluation.save_prediction(root, 'empty',
            {'exploiter': (empty, empty_prob), 'explorer': (empty, empty_prob)}, empty)
        fg = [row for row in evaluation.summarize(rows) if row['class'] == 1]
        self.assertTrue(all(row['hd95_vox'] is None and row['valid_counts']['hd95_vox'] == 0 for row in fg))


if __name__ == '__main__':
    unittest.main(verbosity=2)
