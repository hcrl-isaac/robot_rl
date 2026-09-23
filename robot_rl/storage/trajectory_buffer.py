import torch
from collections.abc import Iterator
from tensordict import TensorDict

from .expert_buffer import ExpertBuffer


def _get_idxs(
    priorities: torch.Tensor,
    num_slices: int,
    seq_length: int,
    bucket_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``(episode, frame)`` indices for sampling consecutive windows from a trajectory buffer."""
    ep_indices = torch.multinomial(priorities, num_slices, replacement=True)
    starts = torch.randint(0, bucket_size - seq_length, (num_slices,), device=priorities.device)
    offsets = torch.arange(seq_length, device=priorities.device)
    seq_indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    ep_flat = ep_indices.unsqueeze(1).expand(num_slices, seq_length).reshape(-1)
    return ep_flat, seq_indices


class TrajectoryBuffer(ExpertBuffer):
    """Subclass of :class:`ExpertBuffer` that stores expert observations in batches of fixed-length trajectories."""

    def __init__(
        self,
        motion_path: str,
        expert_obs_groups: list[str],
        device: str = "cpu",
    ) -> None:
        """Initialize the buffer storage."""
        self.device = device
        self.obs_groups = expert_obs_groups

        # motions: obs tensordict with batch shape (num_motions, bucket_size).
        self.motions = torch.load(motion_path, weights_only=False, map_location="cpu").to(device)
        if len(self.motions.shape) != 2:
            raise ValueError(
                "Expected motions batch size to be 2-dimensional (num_motions, bucket_size), but instead got shape "
                f"{self.motions.shape}."
            )
        self.num_motions, self.bucket_size = self.motions.shape
        self.priorities = torch.ones((self.num_motions,), device=device)
        self._eval_order = torch.arange(0, self.num_motions, device=self.device)

        # mode="default" (no CUDA graphs): this samples under the inference_mode rollout, so a
        # reduce-overhead capture would poison the shared cudagraph pool that update() later reuses.
        self._get_idxs = torch.compile(_get_idxs, mode="default")

        print(f"[INFO] Successfully loaded {self.num_motions} motions with length {self.bucket_size}.")

    def sample(
        self,
        batch_size: int,
        device: str | None = None,
        seq_length: int = 1,
    ) -> tuple[TensorDict, TensorDict]:
        """Sample current and next expert observations from multinomial distribution weighted by priorities.

        When ``seq_length > 1``, samples are returned as ``batch_size // seq_length`` consecutive windows, each of
        length ``seq_length``, drawn from a single motion starting at a random frame. The flat output is ordered so
        that reshaping ``(batch_size, ...) -> (num_slices, seq_length, ...)`` recovers the windows row-wise. With
        ``seq_length = 1`` behavior is equivalent to iid transition sampling.

        Args:
            batch_size: The batch size to sample. Must be divisible by ``seq_length``.
            device: The device to move the output to. Defaults to None, which keeps the observations on the buffer's
                device.
            seq_length: Length of each consecutive window. Must be strictly less than ``bucket_size``.

        Returns:
            A tuple containing the expert obs and next obs as TensorDicts. Shape is (batch_size).
        """
        if batch_size % seq_length != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by seq_length ({seq_length}).")
        if seq_length >= self.bucket_size:
            raise ValueError(f"seq_length ({seq_length}) must be less than bucket_size ({self.bucket_size}).")
        num_slices = batch_size // seq_length
        ep_flat, seq_indices = self._get_idxs(self.priorities, num_slices, seq_length, self.bucket_size)
        return (
            self.motions[ep_flat, seq_indices].to(device),
            self.motions[ep_flat, seq_indices + 1].to(device),
        )

    def sample_states(self, num_envs: int, device: str | None = None) -> dict[str, torch.Tensor]:
        """Sample states for a vectorized environment. Returns a state dictionary.

        See :meth:`get_expert_state` for full state dictionary format.
        """
        ep_indices = torch.multinomial(self.priorities, num_envs, replacement=True)
        motion_indices = torch.randint(0, self.bucket_size, (num_envs,), device=self.device)
        motions = self.motions[ep_indices, motion_indices]
        return self.get_expert_state(motions, device=device)

    def get_batch_motions(
        self,
        mini_batch_size: int,
        device: str | None = None,
    ) -> Iterator[TensorDict]:
        """Sample entire motion trajectories in mini batches.

        Returns iterator containing batched observations as TensorDict with shape (mini_batch_size, bucket_size,
        *obs_size). Note that the final batch may be truncated.
        """
        # Randomize order since rigid body DR is fixed per-environment
        self._eval_order = torch.randperm(self.num_motions, device=self.device)
        for idx in range(0, self.num_motions, mini_batch_size):
            eval_idxs = self._eval_order[idx : idx + mini_batch_size]
            yield self.motions[eval_idxs].to(device)

    @property
    def eval_order(self) -> torch.Tensor:
        """Buffer index of each motion in the order the last :meth:`get_batch_motions` yielded them."""
        return self._eval_order

    def set_priorities(self, priorities: torch.Tensor) -> None:
        """Replace every motion's sampling weight (buffer order) and normalize them to sum to one."""
        if priorities.shape != self.priorities.shape:
            raise ValueError(f"expected {tuple(self.priorities.shape)} priorities, got {tuple(priorities.shape)}")
        self.priorities = priorities.to(self.device, torch.float32) / priorities.sum()

    def state_dict(self) -> dict:
        """Return the per-motion sampling priorities for checkpointing."""
        return {"priorities": self.priorities}

    def load_state_dict(self, state: dict) -> None:
        """Restore the per-motion sampling priorities from a checkpoint."""
        priorities = state["priorities"].to(self.device)
        if priorities.shape != self.priorities.shape:
            raise ValueError(
                f"Loaded expert priorities shape {tuple(priorities.shape)} does not match the current buffer "
                f"{tuple(self.priorities.shape)}; the motion dataset likely differs from the checkpointed run."
            )
        self.priorities = priorities

    def get_expert_state(self, obs: TensorDict, device: str | None = None) -> dict[str, torch.Tensor]:
        """Convert the observations TensorDict into a state dictionary of tensors.

        Dictionary structure and tensor shapes are:

        .. code:: python

            {
                "root_pose": (..., 7),
                "root_velocity": (..., 6),
                "joint_position": (..., num_joints),
                "joint_velocity": (..., num_joints),
            }
        """
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        state = torch.cat(obs_list, dim=-1).to(device)
        num_joints = (state.shape[-1] - 13) // 2
        return {
            "root_pose": state[..., :7],
            "root_velocity": state[..., 7:13],
            "joint_position": state[..., 13 : 13 + num_joints],
            "joint_velocity": state[..., 13 + num_joints :],
        }
