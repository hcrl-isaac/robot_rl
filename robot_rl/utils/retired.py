"""Options removed from the models and algorithms, so cfgs recorded before their removal still rebuild."""

from __future__ import annotations

from typing import Any

RETIRED_OPTIONS: dict[str, Any] = {
    "flatten_proj_dim": 0,
    "attention_pool_memory_dim": 0,
    "attention_pool_latent_layers": 0,
    "attention_pool_memory_norm": False,
    "attention_pool_memory_share": 0.0,
    "spatial_readout": False,
    "backup_entropy": True,
    "critic_on_actor_windows": False,
}
"""Each removed option with the value at which it did nothing."""


def drop_retired_options(cfg: Any) -> Any:
    """Drop retired options left at their no-op value from a recorded cfg tree.

    Args:
        cfg: A cfg as nested dicts and lists.

    Returns:
        The cfg without those keys; a retired option set to anything else raises, since no code here runs it.
    """
    if isinstance(cfg, dict):
        out = {}
        for key, value in cfg.items():
            if key in RETIRED_OPTIONS:
                if value is None or value == RETIRED_OPTIONS[key]:
                    continue
                raise ValueError(f"The recorded cfg sets the removed option {key}={value!r}; no code here runs it.")
            out[key] = drop_retired_options(value)
        return out
    if isinstance(cfg, list):
        return [drop_retired_options(v) for v in cfg]
    return cfg
