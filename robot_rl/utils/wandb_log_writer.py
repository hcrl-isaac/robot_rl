# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

from robot_rl.utils.log_writer import LogWriter

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class WandbLogWriter(SummaryWriter, LogWriter):
    """Summary writer for W&B."""

    def __init__(
        self,
        log_dir: str,
        project_name: str,
        run_name: str | None = None,
        group: str | None = None,
        num_envs: int = 1,
        sim_time_per_iter: float = 0.0,
        shared: bool = False,
        log_videos_async: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize a W&B run for logging."""
        if wandb is None:
            raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.")
        super().__init__(log_dir, flush_secs=10)

        # Get the run name
        run_name = run_name or os.path.split(log_dir)[-1]

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        self.shared = shared
        self.num_envs = num_envs
        self.sim_time_per_iter = sim_time_per_iter

        settings = wandb.Settings(start_method="thread")
        tags = list(tags or [])
        if self.shared:
            settings.x_label = "main"
            settings.mode = "shared"
            settings.x_primary = True
        if log_videos_async:
            tags.append("log_videos_async")

        # Initialize wandb
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            group=group or None,
            config={"log_dir": log_dir},
            settings=settings,
            tags=tags,
        )

        # Define custom metrics
        self.run.define_metric("*", step_metric="local_step")  # global step (custom defined for async video logging)
        self.run.define_metric("*", step_metric="env_step")  # env step (step * num_envs)
        if sim_time_per_iter > 0.0:
            # simulated seconds summed over envs: the only x axis comparable across decision rates
            self.run.define_metric("*", step_metric="sim_time_s")

        # Publish the W&B run identity so out-of-process logger can attach to this exact run
        if self.shared:
            try:
                with open(os.path.join(log_dir, "wandb_run.json"), "w") as f:
                    json.dump(
                        {
                            "id": self.run.id,
                            "project": project_name,
                            "entity": entity,
                            "num_envs": num_envs,
                        },
                        f,
                    )
            except OSError:
                pass

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        # Pin _step to the training iteration (also in shared mode) so a resumed run continues at the
        # checkpoint's iteration instead of restarting _step at 0; secondary writers keep their own sequence.
        entry = {tag: scalar_value, "local_step": global_step, "env_step": global_step * self.num_envs}
        if self.sim_time_per_iter > 0.0:
            entry["sim_time_s"] = global_step * self.sim_time_per_iter
        self.run.log(entry, step=global_step)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        self.run.config.update({"train_cfg": train_cfg})
        try:
            self.run.config.update({"env_cfg": env_cfg.to_dict()})  # type: ignore
        except Exception:
            self.run.config.update({"env_cfg": asdict(env_cfg)})  # type: ignore

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        self.run.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        self.run.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            self.run.log(
                {"video": wandb.Video(str(video), format="mp4"), "local_step": it, "env_step": it * self.num_envs},
                step=it,
            )
            self.logged_videos.add(video.name)

    def stop(self) -> None:
        """Finish the active W&B run."""
        self.run.finish()
