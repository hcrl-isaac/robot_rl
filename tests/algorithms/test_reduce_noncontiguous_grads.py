"""Multi-GPU gradient averaging accepts non-contiguous gradients, as a conv encoder's can be."""

import os
import tempfile
import torch
import torch.distributed as dist
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from robot_rl.algorithms.sac import SAC


@pytest.fixture
def single_rank_group() -> Iterator[None]:
    """Open a one-rank gloo process group, so all_reduce runs on CPU."""
    with tempfile.TemporaryDirectory() as d:
        dist.init_process_group("gloo", init_method=f"file://{os.path.join(d, 'store')}", rank=0, world_size=1)
        yield
        dist.destroy_process_group()


def test_sac_reduce_keeps_non_contiguous_grads(single_rank_group: None) -> None:
    """A transposed gradient averages over one rank back to itself."""
    p = torch.nn.Parameter(torch.zeros(3, 4))
    p.grad = torch.arange(12.0).reshape(4, 3).t()
    assert not p.grad.is_contiguous()
    expected = p.grad.clone()
    SAC.reduce_parameters(SimpleNamespace(gpu_world_size=1), [p])
    assert torch.equal(p.grad, expected)
