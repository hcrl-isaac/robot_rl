"""Tests for dropping retired options from recorded cfgs."""

import pytest

from robot_rl.utils.retired import drop_retired_options


def test_retired_options_drop_only_at_their_no_op_value() -> None:
    """A cfg keeps rebuilding when a removed option sat at its no-op value, and fails when it did something."""
    cnn_cfg = {"attention_pool_queries": 4, "flatten_proj_dim": 0, "attention_pool_memory_norm": None}
    cfg = {"actor": {"cnn_cfg": cnn_cfg}}
    assert drop_retired_options(cfg) == {"actor": {"cnn_cfg": {"attention_pool_queries": 4}}}
    with pytest.raises(ValueError, match="critic_on_actor_windows"):
        drop_retired_options({"algorithm": {"critic_on_actor_windows": True}})
