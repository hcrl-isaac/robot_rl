"""The depth-encoder arms: the gradient a model sends into a shared encoder, and random-shift augmentation."""

import math
import torch
from torch import nn

from robot_rl.algorithms.sac import SAC


class _Tiny(nn.Module):
    """Stands in for a CNN encoder: one parameter, so a gradient is easy to read."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image * self.scale).flatten(1)


def _encode(scale: float) -> float:
    """Gradient that reaches the encoder when a model scales its own contribution by ``scale``."""
    enc = _Tiny()
    head = nn.Parameter(torch.zeros(1))  # keeps the graph alive when the encoder path is fully detached
    feats = enc(torch.ones(2, 1, 2, 2))
    if math.isclose(scale, 1.0):
        out = feats
    elif math.isclose(scale, 0.0):
        out = feats.detach()
    else:
        out = scale * feats + (1.0 - scale) * feats.detach()
    (out.sum() + head.sum()).backward()
    return 0.0 if enc.scale.grad is None else enc.scale.grad.item()


def test_encoder_grad_scale_gates_the_gradient() -> None:
    """The scale decides how much of a loss reaches a shared encoder."""
    assert abs(_encode(1.0) - 8.0) < 1e-6, "an unscaled path trains the encoder"
    assert abs(_encode(0.0)) < 1e-12, "a detached path sends no gradient"
    assert abs(_encode(0.1) - 0.8) < 1e-6, "a scaled path sends a tenth"


def test_random_shift_keeps_shape_and_moves_content() -> None:
    """The shift keeps the frame's shape, moves its content, and is a no-op at pad 0."""
    pad = 4
    image = torch.arange(2 * 4 * 6 * 8, dtype=torch.float32).reshape(2, 4, 6, 8)
    batch = type("B", (), {"observations": {"image": image.clone()}, "next_observations": None})()
    sac = SAC.__new__(SAC)
    sac.image_shift_pad = pad
    sac.actor = type("A", (), {"obs_groups_2d": ["image"]})()
    torch.manual_seed(0)
    sac._shift_images(batch)
    shifted = batch.observations["image"]
    assert shifted.shape == image.shape
    assert not torch.allclose(shifted, image), "a shift of up to 4 px should move the content"
    sac.image_shift_pad = 0
    batch.observations["image"] = image.clone()
    sac._shift_images(batch)
    assert torch.allclose(batch.observations["image"], image), "pad 0 is a no-op"
