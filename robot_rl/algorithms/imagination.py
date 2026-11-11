"""Rebuilding the models' observation groups inside a world-model rollout."""

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
    """``"obs"`` (model observation + noise), ``"state"``/``"context"`` (a slice of either) or ``"action"``."""

    source_slice: slice | None = None
    """The term's slice of its source vector (``"obs"``, ``"state"``, ``"context"`` terms)."""

    noise_slice: slice | None = None
    """The term's slice of the group's noise-bank row (``"obs"`` terms)."""

    @property
    def width(self) -> int:
        """Width of the term's block in the group vector."""
        return self.dim * max(self.history_length, 1)


def _term_slices(spec: WorldModelSpec, group: str | None, obs: TensorDict) -> dict[str, slice]:
    """Name to slice of each single-frame term of ``group`` (empty when the spec names no such group)."""
    if group is None:
        return {}
    out, start = {}, 0
    for term in spec.layout(group, obs[group].shape[-1]):
        if term.history_length > 1:
            raise ValueError(f"World-model group '{group}' must be single-frame; '{term.name}' has history.")
        out[term.name] = slice(start, start + term.dim)
        start += term.dim
    return out


class ImaginedObservation:
    """Advances every model's observation groups one imagined step at a time.

    A whole goal group comes from the goal source, and the spec's state and context groups come whole from the
    model state and the episode's context. Other groups are rebuilt term by term, matched by name: a term of the
    spec's ``obs_group`` comes from the observation head plus sensor noise; an ``action_terms`` entry comes from
    the previous imagined action; a term of the state or context group comes from that vector. History terms roll,
    dropping the oldest frame. Noise is drawn from a bank of logged ``noisy term - noise-free term`` differences
    (one bank row per sample per step, so noise stays correlated across terms), which reproduces per-step additive
    noise of any distribution; noise that persists across steps (biases, latency) is not reproduced.
    """

    def __init__(self, spec: WorldModelSpec, obs: TensorDict, groups: list[str]) -> None:
        """Resolve each term of ``groups`` to its source.

        Args:
            spec: The env's world-model spec.
            obs: An observation sample (for group widths).
            groups: Every group the models read, in any order.
        """
        self.spec = spec
        self.groups = list(dict.fromkeys(groups))
        whole = {spec.state_group, spec.context_group, *spec.goal_groups} - {None}
        sources = {
            "obs": _term_slices(spec, spec.obs_group, obs),
            "state": _term_slices(spec, spec.state_group, obs),
            "context": _term_slices(spec, spec.context_group, obs),
        }
        self.terms: dict[str, list[_Term]] = {}
        self.noise_width: dict[str, int] = {}
        for group in self.groups:
            if group in whole:
                continue
            terms, offset, noise = [], 0, 0
            for t in spec.layout(group, obs[group].shape[-1]):
                if t.name in spec.action_terms:
                    terms.append(_Term(offset, t.dim, t.history_length, "action"))
                else:
                    source = next((s for s in ("obs", "state", "context") if t.name in sources[s]), None)
                    if source is None:
                        raise ValueError(
                            f"Term '{group}/{t.name}' has no world-model source: name it in the spec's obs, state or "
                            "context group, list it in action_terms, or make its group a goal group."
                        )
                    sl = sources[source][t.name]
                    if sl.stop - sl.start != t.dim:
                        raise ValueError(f"Term '{t.name}' is {t.dim} wide in '{group}', not in its {source} group.")
                    noise_sl = slice(noise, noise + t.dim) if source == "obs" else None
                    terms.append(_Term(offset, t.dim, t.history_length, source, sl, noise_sl))
                    noise += t.dim if source == "obs" else 0
                offset += t.width
            self.terms[group] = terms
            self.noise_width[group] = noise

    @property
    def reads_obs_head(self) -> bool:
        """Whether any rebuilt term needs the world model's observation head."""
        return any(t.source == "obs" for terms in self.terms.values() for t in terms)

    def noise_bank(self, rows: TensorDict) -> dict[str, torch.Tensor]:
        """Return the logged sensor noise per group: each noisy term's newest frame minus its noise-free copy.

        Args:
            rows: Logged observation rows holding the rebuilt groups and the spec's obs group.

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
                parts.append(newest - rows[self.spec.obs_group][:, t.source_slice])
            bank[group] = torch.cat(parts, dim=-1) if parts else rows[group][:, :0]
        return bank

    def step(
        self,
        current: TensorDict,
        state: torch.Tensor,
        context: torch.Tensor | None,
        obs_bar: torch.Tensor | None,
        action: torch.Tensor,
        goals: TensorDict | None,
        noise: dict[str, torch.Tensor],
    ) -> TensorDict:
        """Return the groups one step later.

        Args:
            current: The groups at ``t``.
            state: The raw (de-normalized) model state at ``t + 1``.
            context: The episode's raw context (None without a context group).
            obs_bar: The predicted noise-free observation at ``t + 1`` (None without an observation head).
            action: The action taken at ``t`` (the next step's previous action).
            goals: The goal groups at ``t + 1``.
            noise: Per group, one noise row per sample (drawn from :meth:`noise_bank`).

        Returns:
            The groups at ``t + 1``.
        """
        spec = self.spec
        vectors = {"obs": obs_bar, "state": state, "context": context}
        out = {}
        for group in self.groups:
            if group in spec.goal_groups:
                out[group] = goals[group]  # type: ignore[index]
            elif group == spec.state_group:
                out[group] = state
            elif group == spec.context_group:
                out[group] = context
            else:
                blocks = []
                for t in self.terms[group]:
                    if t.source == "action":
                        frame = action
                    else:
                        frame = vectors[t.source][..., t.source_slice]  # type: ignore[index]
                        if t.source == "obs":
                            frame = frame + noise[group][..., t.noise_slice]
                    if t.history_length > 1:
                        old = current[group][..., t.offset + t.dim : t.offset + t.width]
                        blocks.append(torch.cat([old, frame], dim=-1))
                    else:
                        blocks.append(frame)
                out[group] = torch.cat(blocks, dim=-1)
        return TensorDict(out, batch_size=current.batch_size, device=current.device)
