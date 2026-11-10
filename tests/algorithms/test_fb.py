"""Offline forward-backward learner: shapes, losses, latent conventions and checkpoint round trip (CPU)."""

import math
import torch

from robot_rl.algorithms.fb import ForwardBackward, ForwardBackwardCfg


def _small_cfg(**kw: object) -> ForwardBackwardCfg:
    return ForwardBackwardCfg(
        z_dim=8, batch_size=32, hidden_dim=32, feature_dim=16, backward_hidden_dim=16, compile=False, **kw
    )


def _batch(n: int = 32, obs_dim: int = 5, action_dim: int = 2) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "obs": torch.randn(n, obs_dim, generator=g),
        "action": torch.rand(n, action_dim, generator=g) * 2 - 1,
        "next_obs": torch.randn(n, obs_dim, generator=g),
    }


def test_update_returns_finite_diagnostics_and_moves_targets() -> None:
    """Update returns finite diagnostics and moves targets."""
    torch.manual_seed(0)
    agent = ForwardBackward(5, 2, _small_cfg(), device="cpu")
    before = [p.clone() for p in agent.target_forward_map.parameters()]
    for _ in range(3):
        stats = agent.update(_batch())
    assert all(torch.isfinite(v).all() for v in stats.values())
    assert {"Loss/fb", "Loss/orthonormality", "Loss/actor", "FB/q"} <= set(stats)
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.target_forward_map.parameters(), strict=True))
    assert agent.num_updates == 3


def test_backward_and_latents_live_on_the_sqrt_d_sphere() -> None:
    """Backward and latents live on the sqrt d sphere."""
    agent = ForwardBackward(5, 2, _small_cfg(), device="cpu")
    batch = _batch()
    radius = math.sqrt(8)
    assert torch.allclose(agent.backward_map(batch["obs"]).norm(dim=-1), torch.full((32,), radius), atol=1e-4)
    assert torch.allclose(agent.sample_z(32, batch["next_obs"]).norm(dim=-1), torch.full((32,), radius), atol=1e-4)
    z = agent.infer_z(batch["next_obs"], torch.rand(32))
    assert abs(z.norm().item() - radius) < 1e-4
    raw = agent.infer_z(batch["next_obs"], torch.rand(32), normalize=False)
    assert raw.shape == (8,)


def test_actions_are_bounded_and_checkpoint_round_trips() -> None:
    """Actions are bounded and checkpoint round trips."""
    agent = ForwardBackward(5, 2, _small_cfg(), device="cpu")
    agent.update(_batch())
    obs = torch.randn(10, 5) * 10
    z = agent.sample_z(1)[0]
    a = agent.act(obs, z)
    assert a.shape == (10, 2) and a.abs().max() <= 1.0
    clone = ForwardBackward.from_state_dict(agent.state_dict(), device="cpu")
    assert torch.allclose(clone.act(obs, z), a)
    assert clone.num_updates == agent.num_updates
