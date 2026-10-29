"""Rebuilding a policy's observation groups inside a world-model rollout."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from tensordict import TensorDict

from robot_rl.env import WorldModelSpec


@dataclass
class _Term:
    """Where one term of an imagined group comes from."""

    offset: int
    """Start of the term's block in the group vector."""

    dim: int
    """Width of one frame."""

    history_length: int
    """Stacked frames (0 = none)."""

    source: str
    """``"obs"`` (world-model observation + noise), ``"action"`` (previous action) or ``"goal"`` (logged)."""

    obs_slice: slice | None = None
    """The term's slice of the world-model observation (``"obs"`` terms)."""

    noise_slice: slice | None = None
    """The term's slice of the group's noise-bank row (``"obs"`` terms)."""

    @property
    def width(self) -> int:
        """Width of the term's block in the group vector."""
        return self.dim * max(self.history_length, 1)


class ImaginedObservation:
    """Advances a policy's observation groups one imagined step at a time.

    Each term is matched by name: a term of the spec's ``obs_group`` comes from the observation head plus sensor
    noise, an ``action_terms`` entry from the previous imagined action, and a whole goal group from the logged
    window. History terms roll, dropping the oldest frame. Noise is drawn from a bank of logged
    ``noisy term - noise-free term`` differences, so whatever noise model the env applies is reproduced.
    """

    def __init__(self, spec: WorldModelSpec, obs: TensorDict, groups: list[str]) -> None:
        """Resolve each term of ``groups`` to its source.

        Args:
            spec: The env's world-model spec.
            obs: An observation sample (for group widths).
            groups: The policy's observation groups, in the order its model concatenates them.
        """
        self.spec = spec
        self.groups = list(groups)
        obs_terms: dict[str, slice] = {}
        if spec.obs_group is not None:
            start = 0
            for term in spec.layout(spec.obs_group, obs[spec.obs_group].shape[-1]):
                if term.history_length > 1:
                    raise ValueError(f"The world-model obs group must be single-frame; '{term.name}' has history.")
                obs_terms[term.name] = slice(start, start + term.dim)
                start += term.dim
        self.terms: dict[str, list[_Term]] = {}
        self.noise_width: dict[str, int] = {}
        for group in self.groups:
            if group in spec.goal_groups:
                continue
            terms, offset, noise = [], 0, 0
            for t in spec.layout(group, obs[group].shape[-1]):
                if t.name in spec.action_terms:
                    terms.append(_Term(offset, t.dim, t.history_length, "action"))
                elif t.name in obs_terms:
                    sl = obs_terms[t.name]
                    if sl.stop - sl.start != t.dim:
                        raise ValueError(f"Term '{t.name}' is {t.dim} wide in '{group}' but not in the obs group.")
                    terms.append(_Term(offset, t.dim, t.history_length, "obs", sl, slice(noise, noise + t.dim)))
                    noise += t.dim
                else:
                    raise ValueError(
                        f"Actor term '{group}/{t.name}' has no world-model source: add it to the spec's obs group, "
                        "list it in action_terms, or make its group a goal group."
                    )
                offset += t.width
            self.terms[group] = terms
            self.noise_width[group] = noise

    def noise_bank(self, rows: TensorDict) -> dict[str, torch.Tensor]:
        """Return the logged sensor noise per group: each noisy term's newest frame minus its noise-free copy.

        Args:
            rows: Logged observation rows holding the policy groups and the spec's obs group.

        Returns:
            Per group, a ``(num_rows, noise_width)`` bank.
        """
        bank = {}
        for group, terms in self.terms.items():
            parts = []
            for t in terms:
                if t.source != "obs":
                    continue
                newest = rows[group][:, t.offset + t.dim * (max(t.history_length, 1) - 1) : t.offset + t.width]
                parts.append(newest - rows[self.spec.obs_group][:, t.obs_slice])
            bank[group] = torch.cat(parts, dim=-1) if parts else rows[group][:, :0]
        return bank

    def step(
        self,
        current: TensorDict,
        obs_bar: torch.Tensor,
        action: torch.Tensor,
        goals: TensorDict,
        noise: dict[str, torch.Tensor],
    ) -> TensorDict:
        """Return the policy groups one step later.

        Args:
            current: The policy groups at ``t``.
            obs_bar: The predicted noise-free observation at ``t + 1``.
            action: The action taken at ``t`` (the next step's previous action).
            goals: The logged goal groups at ``t + 1``.
            noise: Per group, one noise row per sample (drawn from :meth:`noise_bank`).

        Returns:
            The policy groups at ``t + 1``.
        """
        out = {}
        for group in self.groups:
            if group in self.spec.goal_groups:
                out[group] = goals[group]
                continue
            blocks = []
            for t in self.terms[group]:
                frame = action if t.source == "action" else obs_bar[..., t.obs_slice] + noise[group][..., t.noise_slice]
                if t.history_length > 1:
                    old = current[group][..., t.offset + t.dim : t.offset + t.width]
                    blocks.append(torch.cat([old, frame], dim=-1))
                else:
                    blocks.append(frame)
            out[group] = torch.cat(blocks, dim=-1)
        return TensorDict(out, batch_size=current.batch_size, device=current.device)
