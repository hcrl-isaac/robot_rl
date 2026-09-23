from __future__ import annotations

import contextlib
import os
import time
import torch
from datetime import timedelta
from typing import Any

from robot_rl.algorithms import FbCpr
from robot_rl.env import URLVecEnv
from robot_rl.models import MLPModel
from robot_rl.utils import check_nan, demote_old_checkpoint, resolve_callable
from robot_rl.utils.export import bake_live_normalizer, save_jit, save_onnx
from robot_rl.utils.logger import Logger


class OffPolicyRunner:
    """Off-policy runner for reinforcement learning algorithms."""

    alg: FbCpr
    """The actor-critic algorithm."""

    def __init__(
        self,
        env: URLVecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        inference: bool = False,
    ) -> None:
        """Construct the runner, algorithm, and logging stack.

        ``inference=True`` skips loading the expert motion buffer, so play/visualization can construct the
        policy without the disk load.
        """
        self.cfg = train_cfg
        self.device = device
        self.env = env

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[FbCpr] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device, inference=inference)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self._resumed = False

    def learn(self, num_learning_iterations: int, **kwargs: Any) -> None:
        """Run the learning loop: per iteration, collect env steps, then run agent updates, then log/save."""
        is_url = hasattr(self.alg, "expert_buffer")
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        collect_steps = self.cfg.get("num_steps_per_env", 1)

        # Resolve the update cadence and initial observations. The seed phase (warm up before policy
        # updates begin) is a shared runner-level knob for both the URL and non-URL branches.
        seed_until = start_it + self.cfg["num_seed_steps_per_env"]
        warmup = self.cfg.get("resume_warmup_steps_per_env", 0) if self._resumed else 0
        if warmup:
            # a checkpoint carries no replay buffer: refill it with the loaded policy before updating, on
            # top of the requested learning iterations; random seeding is for an untrained policy
            seed_until = start_it + warmup
            total_it += warmup
            if hasattr(self.alg, "num_seed_steps_per_env"):
                self.alg.num_seed_steps_per_env = -1
            print(f"[INFO] resume warm-up: {warmup} iterations of collection before updates resume")

        def update_gate(it: int) -> bool:
            return it > seed_until

        if is_url:
            num_updates = self.cfg["algorithm"]["num_agent_updates"]
            # Attach the expert buffer, then re-reset so the initial state is RSI'd from the expert buffer
            # rather than the default-pose state from the env wrapper's first reset (ran before attach).
            self.env.set_expert_buffer(self.alg.expert_buffer)
            obs, _ = self.env.reset()
            obs = obs.to(self.device)
            self.env.train_mode()
        else:
            num_updates = 1
            obs = self.env.get_observations().to(self.device)

        # Switch models to train mode (for dropout etc.) and sync parameters across ranks
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        eval_time = 0.0
        collect_time = 0.0
        learn_time = 0.0
        check_for_nan = self.cfg.get("check_for_nan", True)

        with self._get_profile_context() as prof:
            for it in range(start_it, total_it):
                with torch.inference_mode(), torch.profiler.record_function("rollout"):
                    # Run evaluation (URL only; skip_eval bypasses it entirely -- debug-only speed-up)
                    eval_extras = None
                    if (
                        is_url
                        and not self.cfg["algorithm"].get("skip_eval", False)
                        and (it - start_it) % self.cfg["algorithm"]["eval_interval"] == 0
                    ):
                        # Eval runs on rank 0 only (mutates the expert buffer once); other ranks skip
                        # and wait at the barrier below so update()'s all-reduces stay in lockstep.
                        if self.gpu_global_rank == 0:
                            # Save and clear all logging buffers (environments will reset after eval)
                            self.logger.reset_all_envs()
                            # Run evaluation
                            start = time.time()
                            with torch.profiler.record_function("eval"):
                                eval_extras = self.alg.eval(self.env)
                            stop = time.time()
                            eval_time += stop - start

                            # reset env and rollout state (only rank 0's env was perturbed)
                            obs, _ = self.env.reset()
                            obs = obs.to(self.device)
                            self.alg.reset_rollout_state()
                        if self.is_distributed:
                            torch.distributed.barrier()
                            # Eval just rewrote the expert buffer's priorities on rank 0; mirror
                            # them so every rank keeps an identical expert-sampling distribution.
                            priorities = self.alg.expert_buffer.priorities.to(self.device)
                            torch.distributed.broadcast(priorities, src=0)
                            self.alg.expert_buffer.priorities.copy_(priorities)

                    # Collect environment steps
                    start = time.time()
                    for _ in range(collect_steps):
                        actions = self.alg.act(obs)
                        with torch.profiler.record_function("env_step"):
                            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        if check_for_nan:
                            check_nan(obs, rewards, dones)
                        obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        self.logger.process_env_step(
                            rewards, dones, extras, intrinsic_rewards=getattr(self.alg, "intrinsic_rewards", None)
                        )
                    stop = time.time()
                    collect_time += stop - start
                    start = stop

                # Run agent updates
                loss_extras: list[dict] = []
                algo_extras: list[dict] = []
                if update_gate(it):
                    if hasattr(self.alg, "compute_gammas"):
                        with torch.inference_mode():
                            self.alg.compute_gammas()
                    with torch.profiler.record_function("update"):
                        for _ in range(num_updates):
                            out = self.alg.update()
                            loss_dict, algo_dict = out if isinstance(out, tuple) else (out, {})
                            loss_extras.append(loss_dict)
                            if algo_dict:
                                algo_extras.append(algo_dict)
                self.logger.process_update_extras(
                    eval_extras=eval_extras, loss_extras=loss_extras, algo_extras=algo_extras
                )

                stop = time.time()
                learn_time += stop - start
                self.current_learning_iteration = it

                # Log information
                if it % self.cfg.get("log_interval", 1) == 0:
                    log_info = self.alg.log_info() if hasattr(self.alg, "log_info") else {}
                    self.logger.log(
                        it=it,
                        start_it=start_it,
                        total_it=total_it,
                        collect_time=collect_time,
                        learn_time=learn_time,
                        eval_time=eval_time,
                        **log_info,
                    )
                    eval_time = 0.0
                    collect_time = 0.0
                    learn_time = 0.0

                # Save model
                if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore
                    demoted = demote_old_checkpoint(
                        self.alg,
                        self.logger.log_dir,
                        it,
                        self.cfg.get("keep_full_checkpoints"),
                        self.cfg["save_interval"],
                    )
                    if demoted is not None:  # re-upload so the logger's live-sync replaces the full remote copy
                        self.logger.save_model(os.path.join(self.logger.log_dir, f"model_{demoted}.pt"), demoted)

                if prof is not None:
                    prof.step()

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def _get_profile_context(self) -> contextlib.AbstractContextManager[torch.profiler.profile | None]:
        """Build a profiler context manager.

        Returns a :class:`contextlib.nullcontext` (yielding ``None``) if ``cfg["profile"]`` is False, otherwise a
        :class:`torch.profiler.profile` configured from the ``profile_*`` cfg entries.
        """
        if not self.cfg.get("profile", False):
            return contextlib.nullcontext()
        profile_wait = self.cfg.get("profile_wait", 15)
        profile_warmup = self.cfg.get("profile_warmup", 2)
        profile_active = self.cfg.get("profile_active", 3)
        profile_dir = os.path.join(self.logger.log_dir, "profile") if self.logger.log_dir else "profile"
        logger = self.logger

        def _on_trace_ready(prof: torch.profiler.profile) -> None:
            os.makedirs(profile_dir, exist_ok=True)
            trace_path = os.path.join(profile_dir, "trace.json")
            summary_path = os.path.join(profile_dir, "summary.txt")
            prof.export_chrome_trace(trace_path)
            summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            with open(summary_path, "w") as f:
                f.write(summary)
            print(f"[PROFILE] Trace exported to {trace_path}")
            print(summary)
            writer = getattr(logger, "writer", None)
            if writer is not None and hasattr(writer, "save_file"):
                writer.save_file(trace_path)
                writer.save_file(summary_path)
                print("[PROFILE] Trace and summary uploaded to logger.")

        print(
            f"[PROFILE] Will record {profile_active} iters after {profile_wait} wait + {profile_warmup} warmup."
            f" Output dir: {profile_dir}"
        )
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=profile_wait, warmup=profile_warmup, active=profile_active, repeat=1),
            on_trace_ready=_on_trace_ready,
            record_shapes=False,
            with_stack=False,
        )

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used.

        Atomic via ``torch.save`` to ``path + ".tmp"`` then ``os.replace`` -- so a concurrent
        reader (e.g. the out-of-process video logger) never sees a partial multi-GB write.
        Mirrors :meth:`OnPolicyRunner.save`.
        """
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        # Persist cumulative env-steps so a resume reconstructs the curriculum clock at the same
        # sample budget regardless of this run's env/GPU count (see load()).
        saved_dict["env_step"] = int(self.env.unwrapped.common_step_counter) * self.env.num_envs * self.gpu_world_size
        saved_dict["infos"] = infos
        tmp_path = path + ".tmp"
        torch.save(saved_dict, tmp_path)
        os.replace(tmp_path, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for the load; defaults to the runner's device
                (torch.load's own default restores to the SAVED device).
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location or self.device)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
            self._resumed = True
            # Restore the curriculum clock from the cumulative env-step count / this run's effective
            # env count, so a resume with a different env/GPU count doesn't jump the curriculum fraction.
            effective_envs = self.env.num_envs * self.gpu_world_size
            env_step = loaded_dict.get("env_step")
            if env_step is not None:
                self.env.unwrapped.common_step_counter = round(env_step / effective_envs)  # type: ignore
            else:
                self.env.unwrapped.common_step_counter = self.current_learning_iteration * self.cfg["num_steps_per_env"]  # type: ignore
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the actor to a Torch JIT file (baking FbCpr's external obs normalizer in when present)."""
        save_jit(self._export_model().as_jit(), path, filename)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the actor to an ONNX file (baking FbCpr's external obs normalizer in when present)."""
        save_onnx(self._export_model().as_onnx(verbose), path, filename, verbose)

    def _export_model(self) -> MLPModel:
        """Build the export-ready actor: FbCpr normalizes externally (bake it in); SAC normalizes in-model."""
        normalizer = getattr(self.alg, "obs_normalizer", None)
        if normalizer is not None:
            return bake_live_normalizer(self.alg.get_policy(), normalizer).to("cpu")
        return self.alg.get_policy().to("cpu")

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration: the local rank names a device, so a group may sit on any
        # subset of a node's cards (rank_offset.sh) as long as each one exists
        if self.gpu_local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' names no device here ({torch.cuda.device_count()} visible)."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Long timeout: covers the rank-0-only motion eval (~15 min) while other ranks wait at a
        # barrier, which NCCL's default 10-min watchdog would otherwise kill.
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
            timeout=timedelta(hours=2),
        )
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
