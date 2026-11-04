"""Explicit roles and the frozen F02 LA5 configuration contract."""
from copy import deepcopy

ROLES = ('exploiter', 'explorer')
ARCHITECTURES = {'exploiter': 'V', 'explorer': 'U'}


def normalize_config(config, allow_smoke=False):
    config = deepcopy(config)
    if config.get('recipe') != 'F02':
        raise ValueError('Choose the F02 configuration explicitly')
    expected = dict(dataset='LA', labeled_rate=20, nclass=2, image_channel=1,
                    role_architectures=ARCHITECTURES, exploiter_filters=16, explorer_filters=16,
                    lr=0.01, batch_size=4, crop_size=[112, 112, 80], max_iterations=6000,
                    validation_interval=250, eval_patch=[112, 112, 80], eval_stride_xy=18,
                    eval_stride_z=4, consistency_weight_max=1.0, consistency_ramp_iters=1500,
                    accumulation_steps=2, rvp_prob=0.7, rvp_altitude_jitter=7.5, rvp_enabled=True)
    expected_adp = dict(start_epoch=10, lr=0.0001, noise=1e-5, grad_clip=1.0,
                        eps=0.0075, loss_weight=0.4, ema_decay=0.999,
                        sample_batch_size=20480, layer_dims=512, num_blocks=8, clamping=2.0)
    if allow_smoke and config.get('smoke') is True:
        expected.update(max_iterations=2)
        expected_adp.update(start_epoch=0)
    elif config.get('smoke'):
        raise ValueError('A smoke configuration is not a formal recipe')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Unexpected F02 setting: '+key)
    if config.get('adp') != expected_adp:
        raise ValueError('Unexpected F02 ADP configuration')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    return config


def model_state(checkpoint):
    import torch
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('Expected a nonempty plain state_dict')
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError('Inference accepts branch-best state_dict files, not recovery checkpoints')
    if not all(not value.is_floating_point() or torch.isfinite(value).all().item()
               for value in checkpoint.values()):
        raise ValueError('Nonfinite branch weights')
    return checkpoint
