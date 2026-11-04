import torch

class FlowLoss(torch.nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def forward(self, z, sldj, y=None):
        z = z.reshape((z.shape[0], -1))
        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)
        ll = prior_ll + sldj
        nll = -ll.mean()
        return (nll, prior_ll.mean(), sldj.mean())
