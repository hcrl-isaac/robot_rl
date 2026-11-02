"""The replay dump returns the filled rows of the named groups only."""

import torch
from tensordict import TensorDict

from robot_rl.storage.replay_buffer import ReplayBuffer


def test_stored_observations_filled_rows_and_groups() -> None:
    """Only filled rows and only the named groups come back."""
    num_envs = 4
    obs = TensorDict({"a": torch.zeros(num_envs, 3), "b": torch.zeros(num_envs, 2)}, batch_size=num_envs)
    buf = ReplayBuffer(num_envs, 5, obs, [1], 0, 2)
    for step in range(3):
        t = ReplayBuffer.Transition()
        t.observations = TensorDict(
            {"a": torch.full((num_envs, 3), float(step)), "b": torch.ones(num_envs, 2)}, num_envs
        )
        t.next_observations = t.observations.clone()
        t.actions = torch.zeros(num_envs, 1)
        t.rewards = torch.zeros(num_envs)
        t.dones = torch.zeros(num_envs, 1, dtype=torch.bool)
        t.context = torch.zeros(num_envs, 0)
        t.next_terminated = torch.zeros(num_envs, dtype=torch.uint8)
        buf.add_transitions(t)
    out = buf.stored_observations(["a"])
    assert list(out.keys()) == ["a"]
    assert out.batch_size[0] == len(buf) == 3 * num_envs
    assert sorted(out["a"][:, 0].unique().tolist()) == [0.0, 1.0, 2.0]
