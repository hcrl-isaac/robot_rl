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


def test_a_shared_encoder_stays_trainable_while_the_critics_are_frozen() -> None:
    """The bug that voided the first encoder arms: freezing the critics froze the encoder they share."""
    enc = _Tiny()
    critic_only = nn.Parameter(torch.ones(1))
    sac = SAC.__new__(SAC)
    sac.critic_parameters = [*enc.parameters(), critic_only]
    sac._shared_encoder_param_ids = {id(p) for p in enc.parameters()}

    sac._set_critics_trainable(False)
    assert all(p.requires_grad for p in enc.parameters()), "the shared encoder must stay trainable for the actor"
    assert not critic_only.requires_grad, "the critics' own parameters are frozen during the actor update"

    sac._set_critics_trainable(True)
    assert critic_only.requires_grad


def test_without_sharing_every_critic_parameter_freezes() -> None:
    """Without a shared encoder the freeze covers every critic parameter, as plain SAC expects."""
    own = nn.Parameter(torch.ones(1))
    sac = SAC.__new__(SAC)
    sac.critic_parameters = [own]
    sac._shared_encoder_param_ids = set()
    sac._set_critics_trainable(False)
    assert not own.requires_grad
