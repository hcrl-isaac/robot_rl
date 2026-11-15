"""CPU checks for the batched URLB RND agent."""

import torch

from robot_rl.algorithms.urlb_rnd import UrlbRnd, UrlbRndCfg


def _agent(n: int = 3) -> UrlbRnd:
    return UrlbRnd(n, 5, 2, UrlbRndCfg(hidden_dim=32, rnd_rep_dim=8, batch_size=16), device="cpu")


def test_act_shapes_and_bounds():
    agent = _agent()
    obs = torch.randn(3, 5)
    for uniform in (True, False):
        a = agent.act(obs, uniform=uniform)
        assert a.shape == (3, 2)
        assert a.abs().max() <= 1.0


def test_update_changes_each_agent_independently():
    torch.manual_seed(0)
    agent = _agent()
    w0 = agent.critic.trunk.weight.detach().clone()
    obs, nxt = torch.randn(3, 16, 5), torch.randn(3, 16, 5)
    act = torch.rand(3, 16, 2) * 2 - 1
    # agent 1 gets an all-zero batch: its RND error is still computed from its own batch statistics
    obs[1].zero_()
    out = agent.update(obs, act, torch.full((3, 16), 0.97), nxt)
    assert set(out) == {"Loss/rnd", "Loss/critic", "Loss/actor", "Train/intrinsic_reward"}
    assert all(torch.isfinite(v) for v in out.values())
    moved = (agent.critic.trunk.weight.detach() - w0).flatten(1).norm(dim=1)
    assert (moved > 0).all()


def test_target_network_tracks_critic():
    agent = _agent(1)
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(1.0)
    obs = torch.randn(1, 16, 5)
    agent.update(obs, torch.zeros(1, 16, 2), torch.full((1, 16), 0.97), obs)
    gap = [(p - tp).abs().mean() for p, tp in zip(agent.critic.parameters(), agent.critic_target.parameters(), strict=True)]
    assert all(g > 0 for g in gap)
