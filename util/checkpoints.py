"""Canonical paper names, with explicit compatibility for historical H03 keys."""
from copy import deepcopy


CONFIG_ALIASES = {
    'ddfp': 'adp', 'role_trigger_prob': 'rvp_prob',
    'role_altitude_jitter': 'rvp_altitude_jitter',
    'n_filters1': 'exploiter_filters', 'n_filters2': 'explorer_filters',
    'cons_weight': 'consistency_weight_max', 'rampup_iter': 'consistency_ramp_iters',
}
CHECKPOINT_ALIASES = {
    'model1_state_dict': 'exploiter_state_dict',
    'model2_state_dict': 'explorer_state_dict',
    'teacher1_state_dict': 'exploiter_teacher_state_dict',
    'teacher2_state_dict': 'explorer_teacher_state_dict',
    'nf_model1_state_dict': 'exploiter_density_state_dict',
    'nf_model2_state_dict': 'explorer_density_state_dict',
    'optimizer1_state_dict': 'exploiter_optimizer_state_dict',
    'optimizer2_state_dict': 'explorer_optimizer_state_dict',
    'optimizer_nf1_state_dict': 'exploiter_density_optimizer_state_dict',
    'optimizer_nf2_state_dict': 'explorer_density_optimizer_state_dict',
}


def normalize_config(config):
    """Accept the original JSON names, rejecting conflicting duplicate settings."""
    config = deepcopy(config)
    for old, new in CONFIG_ALIASES.items():
        if old in config:
            value = config.pop(old)
            if new in config and config[new] != value:
                raise ValueError(f'Conflicting configuration keys: {old}, {new}')
            config[new] = value
    if 'tree_mode' in config:
        mode = config.pop('tree_mode')
        if mode not in (0, 1):
            raise ValueError('Historical tree_mode must be 0 or 1')
        enabled = mode == 1
        if 'rvp_enabled' in config and config['rvp_enabled'] != enabled:
            raise ValueError('Conflicting tree_mode and rvp_enabled')
        config['rvp_enabled'] = enabled
    # These historical fields never controlled the active Higra operator.
    config.pop('tree_areathrehold', None)
    config.pop('tree_topoprob', None)
    return config


def add_legacy_checkpoint_aliases(checkpoint):
    """Aliases reference the same objects; registered tensor keys stay untouched."""
    for old, new in CHECKPOINT_ALIASES.items():
        if new in checkpoint:
            if old in checkpoint and checkpoint[old] is not checkpoint[new]:
                raise ValueError(f'Conflicting checkpoint keys: {old}, {new}')
            checkpoint[old] = checkpoint[new]
    checkpoint['checkpoint_schema'] = 'h03_paper_roles_v1_with_legacy_aliases'
    return checkpoint


def model_state(checkpoint, role):
    """Extract a plain state_dict or one H03 role, never renaming tensor keys.

    This helper does not deserialize files. Callers retain control over the
    trusted-file policy. Historical original FULL is not assigned an H03 role.
    """
    import torch

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('Expected a nonempty checkpoint dictionary')
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    if role not in ('exploiter', 'explorer'):
        raise ValueError('Expected an H03 role: exploiter or explorer')
    key = role + '_state_dict'
    old = 'model1_state_dict' if role == 'exploiter' else 'model2_state_dict'
    state = checkpoint.get(key, checkpoint.get(old))
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f'Missing plain model state for {role}')
    if key in checkpoint and old in checkpoint:
        other = checkpoint[old]
        if state.keys() != other.keys() or any(not torch.equal(v, other[k]) for k, v in state.items()):
            raise ValueError(f'Conflicting checkpoint states for {role}')
    return state
