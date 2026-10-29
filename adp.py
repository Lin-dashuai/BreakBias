"""Asymmetric density-guided perturbation (ADP), differentiating NF NLL."""
import torch

def adp_perturb_feature_asymmetric(features, nf_model, loss_fn, eps, direction='descending'):
    b, c, d, h, w = features.size()
    features_reshaped = features.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
    features_reshaped.requires_grad_()
    z, log_jac_det = nf_model(features_reshaped)
    nf_loss, _, _ = loss_fn(z, sldj=log_jac_det)
    grad = torch.autograd.grad(nf_loss, features_reshaped, retain_graph=False)[0]
    grad_norm = torch.norm(grad, p=2, dim=1).view(-1, 1) + 1e-08
    if direction == 'descending':
        delta = eps * grad / grad_norm
    elif direction == 'ascending':
        delta = -eps * grad / grad_norm
    else:
        raise ValueError("Direction must be 'descending' or 'ascending'")
    perturbed_features_reshaped = (features_reshaped + delta).detach()
    perturbed_features = perturbed_features_reshaped.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
    return perturbed_features
