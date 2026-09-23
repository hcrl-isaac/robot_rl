# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Callbacks a runner fires after each checkpoint save, e.g. to evaluate it."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import Any

CheckpointCallback = Callable[[Any, str, int], "dict[str, Any] | None"]
"""``fn(runner, path, it) -> outputs``; returned outputs go to ``alg.apply_eval_outputs``."""

ENV_USED = "env_used"
"""Output key a callback sets when it stepped the training env, so the runner resets it before collecting."""


class CheckpointHooks:
    """Mixin: register checkpoint callbacks and apply what they return on every rank."""

    builtin_eval: bool = True
    """Whether the algorithm's own in-loop eval runs; an external eval manager turns it off."""

    def add_checkpoint_callback(self, fn: CheckpointCallback) -> None:
        """Call ``fn(runner, path, it)`` on the main rank after every checkpoint save.

        Args:
            fn: Returns outputs for the algorithm (e.g. ``{"motion_priorities": ...}``), or None.
        """
        if not hasattr(self, "_checkpoint_callbacks"):
            self._checkpoint_callbacks: list[CheckpointCallback] = []
        self._checkpoint_callbacks.append(fn)

    def _after_checkpoint(self, path: str, it: int) -> bool:
        """Run the callbacks, share their outputs across ranks and apply them.

        Every rank must call this at the same iterations; only the main rank runs the callbacks.

        Returns:
            Whether a callback stepped the training env, so the caller must reset it.
        """
        callbacks = getattr(self, "_checkpoint_callbacks", [])
        # only the main rank registers callbacks, so every rank must still join the broadcast
        if not callbacks and not self.is_distributed:  # type: ignore[attr-defined]
            return False
        outputs: dict[str, Any] = {}
        if self.logger.writer is not None:  # type: ignore[attr-defined]
            for fn in callbacks:
                outputs.update(fn(self, path, it) or {})
        if self.is_distributed:  # type: ignore[attr-defined]
            box = [outputs]
            torch.distributed.broadcast_object_list(box, src=0)
            outputs = box[0]
        env_used = bool(outputs.pop(ENV_USED, False))
        if outputs and hasattr(self.alg, "apply_eval_outputs"):  # type: ignore[attr-defined]
            self.alg.apply_eval_outputs(  # type: ignore[attr-defined]
                {k: torch.as_tensor(v, device=self.device) for k, v in outputs.items()}  # type: ignore[attr-defined]
            )
        return env_used

    def _reset_after_eval(self) -> Any:
        """Reset the env and the algorithm's rollout state after an eval stepped the training env."""
        obs, _ = self.env.reset()  # type: ignore[attr-defined]
        for name in ("reset_rollout_state", "reset"):
            fn = getattr(self.alg, name, None)  # type: ignore[attr-defined]
            if fn is not None:
                fn()
                break
        return obs.to(self.device)  # type: ignore[attr-defined]
