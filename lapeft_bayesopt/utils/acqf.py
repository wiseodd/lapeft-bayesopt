import torch
import torch.distributions as dists


def thompson_sampling(
    f_mean: torch.Tensor, f_var: torch.Tensor, gamma: float = 1.0, random_state: int = 1
) -> torch.Tensor:
    """
    Single objective sampling. Note that this is not the exact Thompson sampling since
    f(x) is sampled independently. This is the price we pay for computational
    efficiency since sampling is O(n^3) where n is the size of the candidate set.
    In BO over molecules, n = num. of molecules in the virtual library.
    Nevertheless, this independent Thompson sampling has similar exploration-exploitation
    mechanism as the exact Thompson sampling.

    Parameters:
    -----------
    f_mean: torch.Tensor
        Shape (B,)

    f_var: torch.Tensor
        Shape (B,)

    gamma: 0 <= float <= 1
        Exploration parameter

    Returns:
    --------
    ts: torch.Tensor
        Shape (B,)
    """
    device = f_mean.device
    generator = torch.Generator(device=device).manual_seed(random_state)
    return f_mean + gamma * f_var.sqrt() * torch.randn(
        1, device=device, generator=generator
    )


def thompson_sampling_multivariate(
    f_mean: torch.Tensor, f_cov: torch.Tensor, gamma: float = 1.0, random_state: int = 1
) -> torch.Tensor:
    """
    Multi objective sampling.

    Parameters:
    -----------
    f_mean: torch.Tensor
        Shape (B, K)

    f_cov: torch.Tensor
        Shape (B, K, K)

    Returns:
    --------
    ts: torch.Tensor
        Shape (B, K)
    """
    assert len(f_mean.shape) == 2 and len(f_cov.shape) == 3
    device = f_mean.device
    generator = torch.Generator(device=device).manual_seed(random_state)
    return f_mean + gamma * torch.einsum(
        "bij,bj->bi",
        torch.linalg.cholesky(f_cov),  # (B, K, K)
        torch.randn(f_mean.shape, device=device, generator=generator),  # (B, K)
    )


def ucb(f_mean: torch.Tensor, f_var: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Single objective upper confidence bound.

    Parameters:
    -----------
    f_mean: torch.Tensor
        Shape (B,)

    f_var: torch.Tensor
        Shape (B,)

    gamma: 0 <= float <= 1, default = 0.1
        Exploration parameter

    Returns:
    --------
    ucb: torch.Tensor
        Shape (B,)
    """
    return f_mean + gamma * torch.sqrt(f_var)


def ei(
    f_mean: torch.Tensor, f_var: torch.Tensor, curr_best_val: float, gamma: float = 0.01
) -> torch.Tensor:
    """
    Single-objective expected improvement.

    Parameters:
    -----------
    f_mean: torch.Tensor
        Shape (B,)

    f_var: torch.Tensor
        Shape (B,)

    curr_best_val: float
        Current best function value

    gamma: float, default = 0.01
        Exploration parameter

    Returns:
    --------
    ei: torch.Tensor
        Shape (B,)
    """
    f_std = torch.sqrt(f_var)
    first = f_mean - curr_best_val - gamma
    z = first / f_std
    secnd = dists.Normal(0, 1).cdf(z)
    third = f_std * dists.Normal(0, 1).log_prob(z).exp()
    return first * secnd + third


def scalarize(y: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Scalarize k objective values in y with linear scalarization under the specified weights
    i.e. return sum_{i=1 to k} weights[i] * y[:, i]

    Parameters:
    -----------
    y: torch.Tensor
        Shape (n, k) or (k,) --- the latter implicitly means n = 1

    weights: torch.Tensor, optional
        Shape (k,). If none, then use uniform weighting: [1/k, ..., 1/k]

    Returns:
    --------
    scalarized_y: torch.Tensor
        Shape (n,)
    """
    assert len(y.shape) == 2 or len(y.shape) == 1
    if len(y.shape) == 1:
        y = y[None, :]
    n, k = y.shape
    if weights is not None:
        assert weights.shape == (k,)
    else:
        weights = torch.ones(k) * 1 / k

    return torch.sum(weights[None, :] * y, dim=1)
