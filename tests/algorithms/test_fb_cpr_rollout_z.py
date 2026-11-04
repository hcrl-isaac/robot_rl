"""FB-CPR rollout latents follow each env's own episode step, and match the shared-step schedule when resets align."""

import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from robot_rl.algorithms.fb_cpr import FbCpr, forward_sliding_mean
from robot_rl.utils.utils import eval_mode

Z_DIM = 4


class _ZBuffer:
    def __len__(self) -> int:
        return 16

    def sample(self, num_envs: int, device: str | None = None) -> torch.Tensor:
        return torch.randn(num_envs, Z_DIM)


class _ExpertBuffer:
    def sample(self, batch_size: int, device: str | None = None, seq_length: int = 1) -> tuple[TensorDict, TensorDict]:
        obs = TensorDict({"x": torch.randn(batch_size, Z_DIM)}, batch_size=batch_size)
        return obs, obs


class _Rollout:
    """The attributes and methods ``FbCpr.update_rollout_z`` uses, with stand-in buffers and backward map."""

    steps_per_z_update = 3
    expert_rollout_length = 5
    expert_rollout_ratio = 0.5
    expert_sequence_length = 2
    z_dim = Z_DIM
    device = "cpu"

    def __init__(self) -> None:
        self.expert_rollout_envs = None
        self.expert_rollout_z = None
        self.z_buffer = _ZBuffer()
        self.expert_buffer = _ExpertBuffer()
        self.obs_normalizer = nn.Identity()

    def backward_map(self, obs: TensorDict) -> torch.Tensor:
        return obs["x"]

    project_z = FbCpr.project_z
    _sample_random_z = FbCpr._sample_random_z
    _sample_expert_windows = FbCpr._sample_expert_windows
    _refresh_expert_windows = FbCpr._refresh_expert_windows
    update_rollout_z = FbCpr.update_rollout_z


def _shared_step_update(self: _Rollout, z: torch.Tensor | None, lengths: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Refresh latents on the schedule before per-env episode steps: every env at the same step."""
    step = int(lengths[0].item())
    if z is None or (step % self.steps_per_z_update == 0 and len(self.z_buffer) == 0):
        z = self._sample_random_z(num_envs)
    elif step % self.steps_per_z_update == 0:
        z = self.z_buffer.sample(num_envs, device=self.device)
    rollout_idx = step % self.expert_rollout_length
    if rollout_idx == 0 or self.expert_rollout_envs is None or self.expert_rollout_z is None:
        num_expert_updates = int(num_envs * self.expert_rollout_ratio)
        self.expert_rollout_envs = torch.randperm(num_envs)[:num_expert_updates]
        _, expert_next_obs = self.expert_buffer.sample(
            num_expert_updates * self.expert_rollout_length, seq_length=self.expert_rollout_length
        )
        with eval_mode(self.obs_normalizer):
            expert_next_obs = self.obs_normalizer(expert_next_obs)
        expert_z = self.backward_map(expert_next_obs).view(num_expert_updates, self.expert_rollout_length, -1)
        expert_z = forward_sliding_mean(expert_z, self.expert_sequence_length, dim=1)
        self.expert_rollout_z = math.sqrt(Z_DIM) * nn.functional.normalize(expert_z, dim=-1)
    z[self.expert_rollout_envs] = self.expert_rollout_z[:, rollout_idx]
    return z


def test_time_out_resets_draw_exactly_the_shared_step_schedule() -> None:
    """With every env reset together, each step's latents are bit-identical to the shared-step schedule's."""
    num_envs, episode = 8, 12
    new, old = _Rollout(), _Rollout()
    z_new = z_old = None
    for step in range(40):
        lengths = torch.full((num_envs,), step % episode, dtype=torch.long)
        torch.manual_seed(step)
        z_new = new.update_rollout_z(z_new, lengths, num_envs).clone()
        torch.manual_seed(step)
        z_old = _shared_step_update(old, z_old, lengths, num_envs).clone()
        assert torch.equal(z_new, z_old), f"step {step}"


def test_envs_reset_apart_follow_their_own_schedules() -> None:
    """An env reset alone gets a fresh latent at its own step 0, and the expert share stays fixed."""
    torch.manual_seed(0)
    num_envs = 8
    rollout = _Rollout()
    lengths = torch.zeros(num_envs, dtype=torch.long)
    z = rollout.update_rollout_z(None, lengths, num_envs).clone()
    num_expert = int(num_envs * rollout.expert_rollout_ratio)
    for step in range(1, 30):
        lengths += 1
        lengths[step % num_envs] = 0  # one env resets each step, so steps diverge
        before = z.clone()
        z = rollout.update_rollout_z(z, lengths, num_envs).clone()
        assert int((rollout.expert_rollout_envs >= 0).sum()) <= num_expert
        reset = step % num_envs
        if reset not in rollout.expert_rollout_envs.tolist():
            assert not torch.equal(z[reset], before[reset])
        unchanged = (lengths % rollout.steps_per_z_update != 0) & (lengths % rollout.expert_rollout_length != 0)
        experts = rollout.expert_rollout_envs[rollout.expert_rollout_envs >= 0]
        unchanged[experts] = False
        assert torch.equal(z[unchanged], before[unchanged])
