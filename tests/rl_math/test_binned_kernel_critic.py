"""Direct unit tests for BinnedKernelCritic and _generate_gaussian_kernel_1d,
isolated from the advantage-estimator wrappers that use them."""
import torch

from longnav.utils.rl_core import BinnedKernelCritic, _generate_gaussian_kernel_1d


def test_gaussian_kernel_1d_properties():
    kernel = _generate_gaussian_kernel_1d(sigma=2.0, kernel_size=9, device=torch.device("cpu"))

    assert kernel.shape == (1, 1, 9)
    flat = kernel.view(-1)
    # Normalized: sums to 1.
    assert torch.isclose(flat.sum(), torch.tensor(1.0), atol=1e-6)
    # Symmetric around the center.
    assert torch.allclose(flat, flat.flip(0), atol=1e-6)
    # Peak at the center (odd kernel_size -> unambiguous center index).
    assert torch.argmax(flat).item() == 4


def test_gaussian_kernel_1d_forces_odd_size():
    kernel = _generate_gaussian_kernel_1d(sigma=1.0, kernel_size=8, device=torch.device("cpu"))
    # Even kernel_size is bumped to odd (9) so there's a well-defined center.
    assert kernel.shape == (1, 1, 9)


def test_binned_kernel_critic_fit_predict_recovers_constant_signal():
    """A constant-return signal should be predicted back as (approximately)
    that same constant everywhere, regardless of feature value -- the
    simplest possible correctness check for the kernel-smoothed lookup
    table, without needing to hand-derive the smoothing math."""
    torch.manual_seed(0)
    features = torch.linspace(0, 10, steps=200)
    returns = torch.full((200,), 3.0)

    critic = BinnedKernelCritic(n_bins=64, device="cpu")
    critic.fit(features, returns, sigma=0.5)

    preds = critic.predict(features)
    assert preds.shape == features.shape
    assert torch.allclose(preds, torch.full_like(preds, 3.0), atol=1e-3)


def test_binned_kernel_critic_loto_excludes_own_trajectory():
    """With Leave-One-Trajectory-Out, a trajectory's own data should not
    contribute to its own prediction -- verified by giving one trajectory an
    extreme outlier return and confirming its LOTO prediction is pulled
    toward the *other* trajectories' values, not its own."""
    torch.manual_seed(0)
    n_per_traj = 50
    features = torch.cat([torch.linspace(0, 5, n_per_traj), torch.linspace(0, 5, n_per_traj)])
    # Trajectory 0: constant return of 100 (outlier). Trajectory 1: constant return of 0.
    returns = torch.cat([torch.full((n_per_traj,), 100.0), torch.full((n_per_traj,), 0.0)])
    traj_ids = torch.cat([torch.zeros(n_per_traj, dtype=torch.long), torch.ones(n_per_traj, dtype=torch.long)])

    critic = BinnedKernelCritic(n_bins=32, device="cpu")
    critic.fit(features, returns, traj_ids=traj_ids, sigma=0.5)

    loto_preds_for_traj0 = critic.predict(features[:n_per_traj], query_traj_ids=traj_ids[:n_per_traj])
    # LOTO prediction for trajectory 0 excludes its own (outlier) data, so it
    # should look like trajectory 1's return (0.0), not trajectory 0's (100.0).
    assert torch.allclose(loto_preds_for_traj0, torch.zeros_like(loto_preds_for_traj0), atol=1.0)
