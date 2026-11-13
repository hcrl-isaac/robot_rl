"""Offline forward-backward (FB) representation learning on flat state vectors.

Implements the FB agent of Touati, Rapin & Ollivier, "Does Zero-Shot Reinforcement Learning Exist?" (ICLR 2023):
the successor measure of ``pi_z`` is modeled as ``M(s, a, s') = F(s, a, z)^T B(s')``, learned by the
measure-valued TD loss on a fixed replay buffer, with an orthonormality loss on ``B`` and a deterministic actor
maximizing ``F(s, a, z)^T z``. The cumulant is indexed by the next state: ``Q_r(s, a, z) = F(s, a, z)^T z_r`` with
``z_r = E[r(s') B(s')]``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass
class ForwardBackwardCfg:
    """Hyperparameters of :class:`ForwardBackward`; defaults are the paper's DMC values."""

    z_dim: int = 50
    discount: float = 0.98
    batch_size: int = 1024
    lr: float = 1e-4
    target_tau: float = 0.01
    ortho_coef: float = 1.0
    mix_ratio: float = 0.5
    policy_noise: float = 0.2
    noise_clip: float = 0.3
    hidden_dim: int = 1024
    feature_dim: int = 512
    backward_hidden_dim: int = 256
    compile: bool = True
    q_loss_coef: float = 0.0
    mass_coef: float = 0.0


def _mlp(dims: list[int], first_norm_tanh: bool, final_relu: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if i == 0 and first_norm_tanh:
            layers += [nn.LayerNorm(dims[i + 1]), nn.Tanh()]
        elif not is_last or final_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _TwoStream(nn.Module):
    """Embeds ``(obs, x)`` and ``(obs, z)`` separately, then maps the concatenation to ``out_dim``."""

    def __init__(self, obs_dim: int, x_dim: int, z_dim: int, out_dim: int, cfg: ForwardBackwardCfg) -> None:
        super().__init__()
        h, f = cfg.hidden_dim, cfg.feature_dim
        self.obs_x = _mlp([obs_dim + x_dim, h, f], first_norm_tanh=True, final_relu=True)
        self.obs_z = _mlp([obs_dim + z_dim, h, f], first_norm_tanh=True, final_relu=True)
        self.head = _mlp([2 * f, h, out_dim], first_norm_tanh=False, final_relu=False)

    def forward(self, obs: torch.Tensor, x: torch.Tensor | None, z: torch.Tensor) -> torch.Tensor:
        a = self.obs_x(obs if x is None else torch.cat([obs, x], dim=-1))
        b = self.obs_z(torch.cat([obs, z], dim=-1))
        return self.head(torch.cat([a, b], dim=-1))


class ForwardMap(nn.Module):
    """Twin forward embeddings ``F_1, F_2(s, a, z)``."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: ForwardBackwardCfg) -> None:
        """Build both forward networks.

        Args:
            obs_dim: State dimension.
            action_dim: Action dimension.
            cfg: Learner configuration.
        """
        super().__init__()
        self.nets = nn.ModuleList([_TwoStream(obs_dim, action_dim, cfg.z_dim, cfg.z_dim, cfg) for _ in range(2)])

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return ``(2, batch, z_dim)``."""
        return torch.stack([net(obs, action, z) for net in self.nets])


class BackwardMap(nn.Module):
    """Backward embedding ``B(s)``, L2-normalized to radius ``sqrt(z_dim)``."""

    def __init__(self, obs_dim: int, cfg: ForwardBackwardCfg) -> None:
        """Build the backward network.

        Args:
            obs_dim: State dimension.
            cfg: Learner configuration.
        """
        super().__init__()
        h = cfg.backward_hidden_dim
        self.net = _mlp([obs_dim, h, h, cfg.z_dim], first_norm_tanh=True, final_relu=False)
        self.radius = math.sqrt(cfg.z_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ``B(obs)`` ``(batch, z_dim)``."""
        return self.radius * F.normalize(self.net(obs), dim=-1)


class Actor(nn.Module):
    """Deterministic latent-conditioned policy ``pi_z(s)`` with ``tanh`` outputs."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: ForwardBackwardCfg) -> None:
        """Build the actor network.

        Args:
            obs_dim: State dimension.
            action_dim: Action dimension.
            cfg: Learner configuration.
        """
        super().__init__()
        self.net = _TwoStream(obs_dim, 0, cfg.z_dim, action_dim, cfg)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return actions in ``[-1, 1]`` ``(batch, action_dim)``."""
        return torch.tanh(self.net(obs, None, z))


class ForwardBackward:
    """Offline FB learner holding ``F``, ``B``, the actor, their targets and optimizers."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: ForwardBackwardCfg, device: str = "cuda") -> None:
        """Build the networks.

        Args:
            obs_dim: State dimension.
            action_dim: Action dimension (actions in ``[-1, 1]``).
            cfg: Hyperparameters.
            device: Torch device.
        """
        self.cfg = cfg
        self.device = device
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.forward_map = ForwardMap(obs_dim, action_dim, cfg).to(device)
        self.backward_map = BackwardMap(obs_dim, cfg).to(device)
        self.actor = Actor(obs_dim, action_dim, cfg).to(device)
        self.target_forward_map = copy.deepcopy(self.forward_map).requires_grad_(False)
        self.target_backward_map = copy.deepcopy(self.backward_map).requires_grad_(False)
        fb_params = list(self.forward_map.parameters()) + list(self.backward_map.parameters())
        fused = str(device).startswith("cuda")
        self.fb_optimizer = torch.optim.Adam(fb_params, lr=cfg.lr, fused=fused)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, fused=fused)
        self._online_params = fb_params
        self._target_params = list(self.target_forward_map.parameters()) + list(self.target_backward_map.parameters())
        self._off_diag = 1.0 - torch.eye(cfg.batch_size, device=device)
        self._off_diag_count = cfg.batch_size * (cfg.batch_size - 1)
        self.num_updates = 0
        self._fb_loss = torch.compile(self._fb_loss_impl) if cfg.compile else self._fb_loss_impl
        self._actor_loss = torch.compile(self._actor_loss_impl) if cfg.compile else self._actor_loss_impl

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Projects latents onto the sphere of radius ``sqrt(z_dim)``."""
        return math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=-1)

    @torch.no_grad()
    def sample_z(self, size: int, next_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Sample training latents: uniform on the sphere, a ``mix_ratio`` share replaced by ``B`` of dataset states.

        Args:
            size: Number of latents.
            next_obs: Dataset states to embed for the mixed share (shuffled here).

        Returns:
            Latents ``(size, z_dim)``.
        """
        z = self.project_z(torch.randn(size, self.cfg.z_dim, device=self.device))
        if next_obs is not None and self.cfg.mix_ratio > 0:
            mix = torch.rand(size, device=self.device) < self.cfg.mix_ratio
            idx = torch.randint(0, next_obs.shape[0], (size,), device=self.device)
            z = torch.where(mix[:, None], self.backward_map(next_obs[idx]), z)
        return z

    def _noisy(self, action: torch.Tensor) -> torch.Tensor:
        """Add clipped Gaussian smoothing noise; the clamp keeps the gradient of the mean (straight-through)."""
        noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
        x = action + noise
        clamped = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return x - x.detach() + clamped.detach()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one FB step and one actor step on a batch of transitions.

        Args:
            batch: ``obs``, ``action`` and ``next_obs`` tensors of ``cfg.batch_size`` rows.

        Returns:
            Detached scalar diagnostics keyed ``<Section>/<name>`` (tensors, so a step needs no host sync).
        """
        obs, action, next_obs = batch["obs"], batch["action"], batch["next_obs"]
        z = self.sample_z(obs.shape[0], next_obs)

        loss, stats = self._fb_loss(obs, action, next_obs, z)
        self.fb_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.fb_optimizer.step()

        actor_loss, q = self._actor_loss(obs, z)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            torch._foreach_lerp_(self._target_params, self._online_params, self.cfg.target_tau)
        self.num_updates += 1
        stats["Loss/actor"] = actor_loss.detach()
        stats["FB/q"] = q.detach()
        return stats

    def _fb_loss_impl(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            next_action = self._noisy(self.actor(next_obs, z))
            target_F = self.target_forward_map(next_obs, z, next_action)
            target_B = self.target_backward_map(next_obs)
            target_M = torch.einsum("nsd,td->nst", target_F, target_B).min(dim=0).values

        Fs = self.forward_map(obs, z, action)
        B = self.backward_map(next_obs)
        Ms = torch.einsum("nsd,td->nst", Fs, B)
        diff = Ms - self.cfg.discount * target_M
        fb_offdiag = 0.5 * (diff.pow(2) * self._off_diag).sum() / self._off_diag_count
        fb_diag = -torch.diagonal(Ms, dim1=1, dim2=2).mean(dim=-1).sum()
        cov = B @ B.T
        orth_offdiag = (cov.pow(2) * self._off_diag).sum() / self._off_diag_count
        orth_diag = -2 * cov.diag().mean()
        loss = fb_offdiag + fb_diag + self.cfg.ortho_coef * (orth_offdiag + orth_diag)
        q_loss = torch.zeros((), device=B.device)
        if self.cfg.q_loss_coef > 0:
            # Scalar Bellman consistency of Q = F^T z with the reward B^T Sigma_B^-1 z implied by the batch's B.
            with torch.no_grad():
                next_q = torch.einsum("nsd,sd->ns", target_F, z).min(dim=0).values
                sigma = B.T @ B / B.shape[0]
                implicit_r = (torch.linalg.solve(sigma, B, left=False) * z).sum(dim=-1)
                target_q = implicit_r + self.cfg.discount * next_q
            q = torch.einsum("nsd,sd->ns", Fs, z)
            q_loss = (q - target_q).pow(2).mean(dim=-1).sum()
            loss = loss + self.cfg.q_loss_coef * q_loss
        mass = torch.einsum("nsd,d->ns", Fs, B.detach().mean(dim=0))
        mass_loss = (mass - 1.0 / (1.0 - self.cfg.discount)).pow(2).mean(dim=-1).sum()
        if self.cfg.mass_coef > 0:
            # The successor measure's total mass is sum_t gamma^t = 1 / (1 - gamma); the TD loss alone leaves it free.
            loss = loss + self.cfg.mass_coef * mass_loss
        with torch.no_grad():
            eye_diff = B.T @ B / B.shape[0] - torch.eye(self.cfg.z_dim, device=B.device)
            stats = {
                "Loss/fb": (fb_offdiag + fb_diag).detach(),
                "Loss/fb_offdiag": fb_offdiag.detach(),
                "Loss/fb_diag": fb_diag.detach(),
                "Loss/orthonormality": (orth_offdiag + orth_diag).detach(),
                "FB/target_M": target_M.mean(),
                "FB/forward_norm": Fs[0].norm(dim=-1).mean().detach(),
                "FB/orth_linf": eye_diff.abs().max(),
                "FB/orth_l2": eye_diff.norm() / math.sqrt(self.cfg.z_dim),
                "FB/measure_mass": mass[0].mean().detach(),
                "Loss/mass": mass_loss.detach(),
                "Loss/q": q_loss.detach(),
            }
        return loss, stats

    def _actor_loss_impl(self, obs: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi = self._noisy(self.actor(obs, z))
        q = torch.einsum("nsd,sd->ns", self.forward_map(obs, z, pi), z).min(dim=0).values
        return -q.mean(), q.mean()

    @torch.no_grad()
    def infer_z(self, next_obs: torch.Tensor, reward: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Estimates the task latent ``z_r = E[r(s') B(s')]`` from reward-labeled states.

        Args:
            next_obs: States ``(n, obs_dim)`` sampled from the training distribution.
            reward: Their rewards ``(n,)``.
            normalize: Project onto the latent sphere (the policy input); ``False`` keeps the scale of ``Q_r``.

        Returns:
            ``(z_dim,)`` latent.
        """
        z = (reward[:, None] * self.backward_map(next_obs)).mean(dim=0)
        return self.project_z(z) if normalize else z

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the deterministic actions of ``pi_z`` (``z`` broadcast over the batch)."""
        return self.actor(obs, z.expand(obs.shape[0], -1))

    def state_dict(self) -> dict:
        """Return everything needed to resume or evaluate."""
        return {
            "cfg": dataclasses.asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "forward_map": self.forward_map.state_dict(),
            "backward_map": self.backward_map.state_dict(),
            "actor": self.actor.state_dict(),
            "target_forward_map": self.target_forward_map.state_dict(),
            "target_backward_map": self.target_backward_map.state_dict(),
            "fb_optimizer": self.fb_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "num_updates": self.num_updates,
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cuda") -> ForwardBackward:
        """Rebuilds a learner saved with :meth:`state_dict`."""
        agent = cls(state["obs_dim"], state["action_dim"], ForwardBackwardCfg(**state["cfg"]), device)
        for name in (
            "forward_map",
            "backward_map",
            "actor",
            "target_forward_map",
            "target_backward_map",
            "fb_optimizer",
            "actor_optimizer",
        ):
            getattr(agent, name).load_state_dict(state[name])
        agent.num_updates = state["num_updates"]
        return agent
