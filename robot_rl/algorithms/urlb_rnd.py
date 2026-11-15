"""URLB's reward-free RND agent (DDPG + random network distillation), batched over independent agents.

Matches the agent that collected the ExORL datasets (url_benchmark ``agent/ddpg.py``, ``agent/rnd.py``): every
network has a leading agent dimension, so one update trains ``num_agents`` independent agents (e.g. one per
dynamics shift) with no shared parameters or data.
"""

from __future__ import annotations

import dataclasses
import math
import torch
from torch import nn


@dataclasses.dataclass
class UrlbRndCfg:
    """Hyperparameters of URLB's state-based DDPG + RND (``ddpg.yaml``, ``rnd.yaml``, ``pretrain.yaml``)."""

    hidden_dim: int = 1024
    rnd_rep_dim: int = 512
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    discount: float = 0.99
    nstep: int = 3
    batch_size: int = 1024
    stddev: float = 0.2
    stddev_clip: float = 0.3
    rnd_scale: float = 1.0
    obs_clip: float = 5.0


class BatchedLinear(nn.Module):
    """``num_agents`` independent linear layers applied to ``(agents, batch, in)`` inputs."""

    def __init__(self, num_agents: int, in_dim: int, out_dim: int) -> None:
        """Initialize orthogonal weights and zero biases (URLB's ``weight_init``).

        Args:
            num_agents: Independent layers.
            in_dim: Input size.
            out_dim: Output size.
        """
        super().__init__()
        w = torch.empty(num_agents, out_dim, in_dim)
        for a in range(num_agents):
            nn.init.orthogonal_(w[a])
        self.weight = nn.Parameter(w.transpose(1, 2).contiguous())
        self.bias = nn.Parameter(torch.zeros(num_agents, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each agent's layer to its own batch."""
        return torch.baddbmm(self.bias, x, self.weight)


class BatchedLayerNorm(nn.Module):
    """Layer norm over the feature dimension with a per-agent affine map."""

    def __init__(self, num_agents: int, dim: int) -> None:
        """Initialize unit scales and zero shifts.

        Args:
            num_agents: Independent norms.
            dim: Feature size.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_agents, 1, dim))
        self.bias = nn.Parameter(torch.zeros(num_agents, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each row, then apply the agent's affine map."""
        return nn.functional.layer_norm(x, x.shape[-1:]) * self.weight + self.bias


class Actor(nn.Module):
    """URLB's state actor: a layer-normed tanh trunk, one hidden layer, tanh mean action."""

    def __init__(self, n: int, obs_dim: int, act_dim: int, hidden: int) -> None:
        """Build the layers for ``n`` agents.

        Args:
            n: Agents.
            obs_dim: Observation size.
            act_dim: Action size.
            hidden: Hidden width.
        """
        super().__init__()
        self.l1, self.ln = BatchedLinear(n, obs_dim, hidden), BatchedLayerNorm(n, hidden)
        self.l2, self.l3 = BatchedLinear(n, hidden, hidden), BatchedLinear(n, hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action in ``[-1, 1]``."""
        h = torch.tanh(self.ln(self.l1(obs)))
        return torch.tanh(self.l3(nn.functional.relu(self.l2(h))))


class Critic(nn.Module):
    """URLB's state critic: a shared layer-normed tanh trunk on (obs, action) and two Q heads."""

    def __init__(self, n: int, obs_dim: int, act_dim: int, hidden: int) -> None:
        """Build the layers for ``n`` agents.

        Args:
            n: Agents.
            obs_dim: Observation size.
            act_dim: Action size.
            hidden: Hidden width.
        """
        super().__init__()
        self.trunk, self.ln = BatchedLinear(n, obs_dim + act_dim, hidden), BatchedLayerNorm(n, hidden)
        self.q1a, self.q1b = BatchedLinear(n, hidden, hidden), BatchedLinear(n, hidden, 1)
        self.q2a, self.q2b = BatchedLinear(n, hidden, hidden), BatchedLinear(n, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q estimates."""
        h = torch.tanh(self.ln(self.trunk(torch.cat([obs, action], dim=-1))))
        return self.q1b(nn.functional.relu(self.q1a(h))), self.q2b(nn.functional.relu(self.q2a(h)))


class RndNet(nn.Module):
    """RND predictor or target: a three-layer ReLU MLP to the representation."""

    def __init__(self, n: int, obs_dim: int, hidden: int, rep: int) -> None:
        """Build the layers for ``n`` agents.

        Args:
            n: Agents.
            obs_dim: Observation size.
            hidden: Hidden width.
            rep: Representation size.
        """
        super().__init__()
        self.l1 = BatchedLinear(n, obs_dim, hidden)
        self.l2 = BatchedLinear(n, hidden, hidden)
        self.l3 = BatchedLinear(n, hidden, rep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the representation of normalized observations."""
        return self.l3(nn.functional.relu(self.l2(nn.functional.relu(self.l1(x)))))


def _sample_action(
    mu: torch.Tensor, std: float, clip: float | None, generator: torch.Generator | None = None
) -> torch.Tensor:
    """URLB's TruncatedNormal.sample: Gaussian noise (optionally clipped), then a straight-through clamp to (-1, 1)."""
    eps = torch.randn(mu.shape, device=mu.device, generator=generator) * std
    if clip is not None:
        eps = eps.clamp(-clip, clip)
    x = mu + eps
    clamped = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return x - x.detach() + clamped.detach()


class UrlbRnd:
    """``num_agents`` independent reward-free DDPG + RND agents updated in one step."""

    def __init__(
        self, num_agents: int, obs_dim: int, act_dim: int, cfg: UrlbRndCfg | None = None, device: str = "cuda"
    ) -> None:
        """Build networks, optimizers and per-agent intrinsic-reward statistics.

        Args:
            num_agents: Independent agents (each has its own networks, optimizer state and intrinsic-reward scale).
            obs_dim: Observation size.
            act_dim: Action size (actions in ``[-1, 1]``).
            cfg: Hyperparameters.
            device: Torch device.
        """
        self.cfg = cfg = cfg or UrlbRndCfg()
        self.n, self.device = num_agents, device
        h = cfg.hidden_dim
        self.actor = Actor(num_agents, obs_dim, act_dim, h).to(device)
        self.critic = Critic(num_agents, obs_dim, act_dim, h).to(device)
        self.critic_target = Critic(num_agents, obs_dim, act_dim, h).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.predictor = RndNet(num_agents, obs_dim, h, cfg.rnd_rep_dim).to(device)
        self.target = RndNet(num_agents, obs_dim, h, cfg.rnd_rep_dim).to(device).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.rnd_opt = torch.optim.Adam(self.predictor.parameters(), lr=cfg.lr)
        # utils.RMS per agent: mean 0, variance 1, count 1e-4
        self.rms_mean = torch.zeros(num_agents, 1, device=device)
        self.rms_var = torch.ones(num_agents, 1, device=device)
        self.rms_count = torch.full((num_agents, 1), 1e-4, device=device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, uniform: bool, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return one exploration action per agent for ``obs`` ``(agents, obs_dim)``; uniform while seeding."""
        mu = self.actor(obs[:, None])[:, 0]
        if uniform:
            return torch.rand(mu.shape, device=mu.device, generator=generator) * 2 - 1
        return _sample_action(mu, self.cfg.stddev, None, generator)

    def _rnd_error(self, obs: torch.Tensor) -> torch.Tensor:
        # Always fp32: the novelty signal is a small difference of O(1) outputs that reduced precision erases.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast(obs.device.type, enabled=False):
                obs = obs.float()
                # BatchNorm1d(affine=False) in train mode: each agent's own batch statistics, then clamp
                mean, var = obs.mean(1, keepdim=True), obs.var(1, unbiased=False, keepdim=True)
                x = ((obs - mean) / torch.sqrt(var + 1e-5)).clamp(-self.cfg.obs_clip, self.cfg.obs_clip)
                return (self.target(x) - self.predictor(x)).square().mean(-1)  # (agents, batch)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

    @torch.no_grad()
    def _intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        err = self._rnd_error(obs)
        bs = err.shape[1]
        delta = err.mean(1, keepdim=True) - self.rms_mean
        total = self.rms_count + bs
        self.rms_var = (
            self.rms_var * self.rms_count + err.var(1, keepdim=True) * bs + delta.square() * self.rms_count * bs / total
        ) / total
        self.rms_mean = self.rms_mean + delta * bs / total
        self.rms_count = total
        return self.cfg.rnd_scale * err / (torch.sqrt(self.rms_var) + 1e-8)

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one URLB update per agent: RND predictor, critic on the intrinsic reward, actor, target averaging.

        Args:
            obs: ``(agents, batch, obs_dim)`` states s_t.
            action: ``(agents, batch, act_dim)`` actions a_t.
            discount: ``(agents, batch)`` n-step discounts (gamma^n with no terminals).
            next_obs: ``(agents, batch, obs_dim)`` states s_{t+n}.
            generator: Noise generator.

        Returns:
            Mean losses and intrinsic reward over agents.
        """
        cfg = self.cfg
        rnd_loss = self._rnd_error(obs).mean(1).sum()
        self.rnd_opt.zero_grad(set_to_none=True)
        rnd_loss.backward()
        self.rnd_opt.step()
        reward = self._intrinsic_reward(obs)  # the critic sees only the one-step intrinsic reward of s_t

        with torch.no_grad():
            next_action = _sample_action(self.actor(next_obs), cfg.stddev, cfg.stddev_clip, generator)
            tq1, tq2 = self.critic_target(next_obs, next_action)
            target = reward[..., None] + discount[..., None] * torch.min(tq1, tq2)
        q1, q2 = self.critic(obs, action)
        mse = nn.functional.mse_loss
        # per-agent mean, summed so each agent's gradient equals its own loss's
        critic_loss = (mse(q1, target, reduction="none") + mse(q2, target, reduction="none")).mean((1, 2)).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        a = _sample_action(self.actor(obs), cfg.stddev, cfg.stddev_clip, generator)
        aq1, aq2 = self.critic(obs, a)
        actor_loss = -torch.min(aq1, aq2).mean((1, 2)).sum()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
                tp.lerp_(p, cfg.critic_target_tau)
        n = self.n
        return {
            "Loss/rnd": rnd_loss.detach() / n,
            "Loss/critic": critic_loss.detach() / n,
            "Loss/actor": actor_loss.detach() / n,
            "Train/intrinsic_reward": reward.mean(),
        }

    @staticmethod
    def nstep_discount(cfg: UrlbRndCfg) -> float:
        """Return the n-step discount with no terminals, gamma^n."""
        return math.pow(cfg.discount, cfg.nstep)
