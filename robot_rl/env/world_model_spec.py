"""The env-side description of a world model's spaces, handed to world-model algorithms by the env wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObsTermLayout:
    """One observation term's place in its group's flat vector."""

    name: str
    """Term name, matched across groups (an actor term is rebuilt from the same-named world-model term)."""

    dim: int
    """Width of one frame of the term."""

    history_length: int = 0
    """Frames the group stacks for this term (0 = no history); stacked oldest first, newest last."""

    @property
    def width(self) -> int:
        """Width of the term's block in the flat group vector."""
        return self.dim * max(self.history_length, 1)


@dataclass
class WorldModelSpec:
    """Which observation groups make up a world model's state, observation, context and goal.

    The env owns this (what the model sees); the algorithm cfg owns how it is modelled.
    """

    state_group: str
    """Group predicted autoregressively by the dynamics model."""

    obs_group: str | None = None
    """Noise-free single-frame copies of the policy's noisy terms (same names and widths, substep stacks included),
    predicted by the observation head."""

    context_group: str | None = None
    """Per-episode simulation context (domain-randomization parameters), constant along a rollout."""

    goal_groups: list[str] = field(default_factory=list)
    """Exogenous groups (e.g. the reference window): never predicted. Read from the env's goal provider when
    ``goal_key_group`` is set, otherwise along the logged trajectory."""

    goal_key_group: str | None = None
    """Group holding each row's goal key (e.g. clip and frame); the env's ``world_model_goals(keys, steps)`` returns
    the goal groups ``steps`` env steps after that key, so imagination needs no logged future."""

    action_terms: list[str] = field(default_factory=list)
    """Actor-observation terms equal to the previous action; imagination fills them with the imagined action."""

    layouts: dict[str, list[ObsTermLayout]] = field(default_factory=dict)
    """Term layout of every observation group, in the env's concatenation order."""

    def layout(self, group: str, width: int) -> list[ObsTermLayout]:
        """Return the term layout of ``group``, or a single anonymous term spanning ``width`` if none was given.

        Args:
            group: Observation group name.
            width: The group's flat width, used to check the layout against the data.

        Returns:
            The group's term layout.
        """
        terms = self.layouts.get(group, [ObsTermLayout(name=group, dim=width)])
        if sum(t.width for t in terms) != width:
            raise ValueError(f"Layout of group '{group}' spans {sum(t.width for t in terms)} values, data has {width}.")
        return terms

    @staticmethod
    def from_dict(cfg: dict) -> WorldModelSpec:
        """Build a spec from its dict form (as a cfg or ``dataclasses.asdict`` produces it)."""
        cfg = dict(cfg)
        layouts = {
            group: [term if isinstance(term, ObsTermLayout) else ObsTermLayout(**term) for term in terms]
            for group, terms in cfg.pop("layouts", {}).items()
        }
        return WorldModelSpec(layouts=layouts, **cfg)
