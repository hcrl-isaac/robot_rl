# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for SAC with a recurrent CNN actor (dummy VecEnv with an image group, no Isaac Sim)."""

import torch
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import SAC
from robot_rl.storage import ReplayBuffer

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 3, 8
IMG = (1, 8, 12)


class _DummyImageVecEnv:
    """A minimal VecEnv with a 1D actor group, a 2D image group, and a privileged critic group."""

    def __init__(self, num_envs: int = NUM_ENVS, num_actions: int = ACT_DIM, done_prob: float = 0.1) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.done_prob = done_prob
        self.cfg = None

    def _obs(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM),
                "image": torch.rand(self.num_envs, *IMG),
                "critic": torch.randn(self.num_envs, OBS_DIM),
                "target": torch.randn(self.num_envs, 3),
                "pixel": torch.cat([torch.rand(self.num_envs, 2), torch.ones(self.num_envs, 1)], dim=1),
                "view": torch.cat([torch.randn(self.num_envs, 3), (torch.rand(self.num_envs, 1) < 0.5).float()], dim=1),
            },
            batch_size=self.num_envs,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor) -> tuple:
        next_obs = self._obs()
        rewards = torch.randn(self.num_envs)
        dones = torch.rand(self.num_envs) < self.done_prob
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool), "time_outs_obs": None}
        return next_obs, rewards, dones, extras


def _make_cfg(obs_normalization: bool = True, aux_obs_group: str | None = None, n_steps: int = 1) -> dict:
    return {
        "num_steps_per_env": 4,
        "obs_groups": {"actor": ["policy", "image"], "critic": ["critic"]},
        "actor": {
            "class_name": "CNNRNNModel",
            "hidden_dims": [32],
            "obs_normalization": obs_normalization,
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "cnn_cfg": {
                "output_channels": [4, 4],
                "kernel_size": 3,
                "stride": 2,
                "activation": "elu",
                "global_pool": "avg",
            },
            "distribution_cfg": {"class_name": "SquashedTanhGaussianDistribution", "init_noise_std": 1.0},
        },
        "critic": {"class_name": "FuseModel", "embedding_dims": [16], "hidden_dims": [32]},
        "algorithm": {
            "class_name": "SAC",
            "replay_buffer_size": 512,
            "mini_batch_size": 8,
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "alpha_learning_rate": 1e-3,
            "auto_alpha": True,
            "alpha": 0.1,
            "tau": 0.1,
            "gamma": 0.99,
            "target_entropy_scale": 1.0,
            "max_grad_norm": 1.0,
            "policy_frequency": 1,
            "n_steps": n_steps,
            "seq_len": 4,
            "burn_in": 2,
            "aux_obs_group": aux_obs_group,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def _build(
    done_prob: float = 0.1, obs_normalization: bool = True, aux_obs_group: str | None = None, n_steps: int = 1
) -> tuple[SAC, _DummyImageVecEnv]:
    env = _DummyImageVecEnv(done_prob=done_prob)
    cfg = _make_cfg(obs_normalization, aux_obs_group, n_steps)
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
    return alg, env


def _collect(alg: SAC, env: _DummyImageVecEnv, steps: int) -> list[torch.Tensor]:
    """Roll out and return the dones of every step."""
    obs = env.get_observations()
    all_dones = []
    for _ in range(steps):
        actions = alg.act(obs)
        next_obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(next_obs, rewards, dones, extras)
        all_dones.append(dones)
        obs = next_obs
    return all_dones


class TestRecurrentSAC:
    """Recurrent actor end to end: stored state, reset on done, windowed update."""

    def test_construct_stores_hidden(self) -> None:
        """A recurrent actor makes the buffer allocate hidden-state storage sized from the RNN."""
        alg, _ = _build()
        assert alg.recurrent
        assert alg.replay_buffer.hidden is not None
        assert alg.replay_buffer.hidden.shape[-1] == 16

    def test_stored_state_is_self_consistent(self) -> None:
        """Stored states chain: the state after obs_t equals the state stored before obs_t+1, and a done zeros it.

        Normalization is off so the recomputation sees the same features the rollout did; with it on, the
        running statistics move between the two and the comparison measures that drift instead.
        """
        torch.manual_seed(0)
        alg, env = _build(done_prob=0.5, obs_normalization=False)
        dones = _collect(alg, env, 6)
        buf = alg.replay_buffer
        n = env.num_envs
        for t in range(5):
            row_t, row_next = t * n, (t + 1) * n
            h_t = buf.hidden[row_t : row_t + n, 0].permute(1, 0, 2)  # (layers, N, H)
            obs_t = buf.observations[row_t : row_t + n]
            with torch.no_grad():
                _, h_after = alg.actor.encode_sequence(obs_t.unsqueeze(0), h_t)
            h_stored_next = buf.hidden[row_next : row_next + n, 0].permute(1, 0, 2)
            done = dones[t]
            assert torch.allclose(h_after[:, ~done], h_stored_next[:, ~done], atol=1e-6)
            if done.any():
                assert torch.all(h_stored_next[:, done] == 0)

    def test_update_runs_and_is_finite(self) -> None:
        """After enough contiguous data, the windowed update runs every loss and returns finite values."""
        torch.manual_seed(0)
        alg, env = _build()
        _collect(alg, env, 12)
        losses, diag = alg.update()
        for key in ("critic_1", "critic_2", "actor", "alpha"):
            assert torch.isfinite(torch.tensor(losses[key])), key
        assert losses["critic_1"] > 0
        for key in (
            "Recurrent/state_staleness",
            "Recurrent/init_carryover",
            "Recurrent/episode_starts_per_window",
            "Recurrent/burn_in_crosses_episode",
            "Recurrent/zero_init_frac",
            "Recurrent/sample_age",
            "Recurrent/hidden_abs_rollout",
            "Critic/q_data",
            "Critic/td_abs",
            "Critic/q_pi_gap",
            "Critic/grad_norm",
            "Actor/logp",
            "Actor/grad_norm_rnn",
        ):
            assert key in diag and torch.isfinite(torch.tensor(diag[key])), key
        assert 0.0 <= diag["Recurrent/burn_in_crosses_episode"] <= 1.0
        assert 0.0 <= diag["Recurrent/zero_init_frac"] <= 1.0
        assert 0.0 < diag["Recurrent/sample_age"] <= 1.0
        assert diag["Recurrent/state_staleness"] >= 0.0
        assert diag["Actor/grad_norm"] > 0.0 and diag["Actor/grad_norm_rnn"] > 0.0

    def test_staleness_is_zero_before_any_weight_change(self) -> None:
        """With unchanged weights the re-derived burn-in state equals the stored one, so staleness reads ~0."""
        torch.manual_seed(0)
        alg, env = _build(obs_normalization=False)
        _collect(alg, env, 12)
        batch = alg.replay_buffer.sample_sequences(alg.seq_len, alg.burn_in, alg.device)
        seen: dict[str, float] = {}
        with torch.no_grad():
            _, h0 = alg.actor.encode_sequence(
                batch.observations[: batch.burn_in], batch.init_hidden, batch.resets[: batch.burn_in]
            )
            alg._recurrent_state_diagnostics(batch, h0, lambda k, v: seen.__setitem__(k, float(v)))
        assert seen["Recurrent/state_staleness"] < 1e-4

    def test_update_before_window_exists_is_a_noop(self) -> None:
        """With fewer rows than burn_in + seq_len, no update is attempted and nothing crashes."""
        alg, env = _build()
        _collect(alg, env, 2)
        step_before = alg.update_step
        losses, _ = alg.update()
        assert alg.update_step == step_before
        assert losses["critic_1"] == 0.0

    def test_sequence_act_matches_stepwise_rollout(self) -> None:
        """Encoding a window in one call equals stepping the rollout RNN through it one step at a time."""
        torch.manual_seed(0)
        alg, env = _build()
        obs_seq = [env.get_observations() for _ in range(5)]
        alg.actor.reset()
        with torch.no_grad():
            stepwise = [alg.actor.get_latent(obs) for obs in obs_seq]
            window = TensorDict.stack(obs_seq, dim=0)
            latent, _ = alg.actor.encode_sequence(window, None)
        assert torch.allclose(latent, torch.stack(stepwise), atol=1e-6)

    def test_window_resets_match_fresh_episodes(self) -> None:
        """A window with a reset at step k encodes steps k.. exactly as a fresh rollout from zeros would."""
        torch.manual_seed(0)
        alg, env = _build()
        obs_seq = [env.get_observations() for _ in range(6)]
        window = TensorDict.stack(obs_seq, dim=0)
        resets = torch.zeros(6, NUM_ENVS)
        resets[3] = 1.0
        with torch.no_grad():
            latent, _ = alg.actor.encode_sequence(window, None, resets)
            fresh, _ = alg.actor.encode_sequence(TensorDict.stack(obs_seq[3:], dim=0), None)
        assert torch.allclose(latent[3:], fresh, atol=1e-6)
        assert not torch.allclose(latent[:3], fresh[:3], atol=1e-3)

    def test_encode_step_from_per_step_states_matches_sequence(self) -> None:
        """Stepping obs[t+1] from the state after obs[t] reproduces the sequence latent at t+1."""
        torch.manual_seed(0)
        alg, env = _build()
        obs_seq = [env.get_observations() for _ in range(5)]
        window = TensorDict.stack(obs_seq, dim=0)
        resets = torch.zeros(5, NUM_ENVS)
        resets[2] = 1.0
        with torch.no_grad():
            latent, _, states = alg.actor.encode_sequence_with_states(window, None, resets)
            for t in range(4):
                if resets[t + 1].any():
                    continue
                stepped = alg.actor.encode_step(obs_seq[t + 1], states[t])
                assert torch.allclose(stepped, latent[t + 1], atol=1e-6)

    def test_head_sees_features_and_memory(self) -> None:
        """The head input is the current feature vector followed by the recurrent state."""
        alg, env = _build()
        obs = env.get_observations()
        alg.actor.reset()
        with torch.no_grad():
            latent = alg.actor.get_latent(obs)
            feats = alg.actor._features(obs)
        assert latent.shape[-1] == feats.shape[-1] + 16
        assert torch.allclose(latent[:, : feats.shape[-1]], feats)

    def test_critic_bootstrap_state_matches_rollout(self) -> None:
        """Stepping the stored pre-step state through obs_t gives the state stored before obs_t+1 (no done)."""
        torch.manual_seed(0)
        alg, env = _build(done_prob=0.0, obs_normalization=False)
        _collect(alg, env, 4)
        buf = alg.replay_buffer
        n = env.num_envs
        batch = buf.sample_mini_batch()
        rows = buf._indices.clone()
        with torch.no_grad():
            h_next = alg.actor.step_state(batch.observations, batch.hidden)
        for i, row in enumerate(rows.tolist()):
            if row + n >= len(buf):
                continue
            assert torch.allclose(h_next[:, i], buf.hidden[row + n, 0], atol=1e-6)

    def test_nstep_batch_discounts_by_aggregated_horizon(self) -> None:
        """An n-step mini-batch bootstraps with gamma^k, k the steps summed into each sample (1..n_steps)."""
        torch.manual_seed(0)
        alg, env = _build(done_prob=0.1, n_steps=5)
        _collect(alg, env, 12)
        batch = alg.replay_buffer.sample_mini_batch(alg.device)
        assert batch.effective_n_steps is not None
        k = batch.effective_n_steps.view(-1).float()
        assert torch.all(k >= 1) and torch.all(k <= 5) and k.max() > 1
        assert torch.allclose(alg._bootstrap_discount(batch), alg.gamma**k)
        losses, diag = alg.update()
        assert torch.isfinite(torch.tensor(losses["critic_1"])) and torch.isfinite(
            torch.tensor(diag["Critic/q_target"])
        )

    def test_nstep_bootstrap_state_is_the_last_steps_stored_state(self) -> None:
        """An n-step sample bootstraps from the pre-step state stored at its n-th step, not its first."""
        torch.manual_seed(0)
        alg, env = _build(done_prob=0.0, obs_normalization=False, n_steps=3)
        _collect(alg, env, 12)
        buf = alg.replay_buffer
        n = env.num_envs
        batch = buf.sample_mini_batch()
        assert batch.hidden is not None and batch.bootstrap_hidden is not None
        obs_all = buf.observations["policy"][: len(buf)]
        for i in range(8):
            row = (obs_all == batch.bootstrap_observations["policy"][i]).all(dim=-1).nonzero()[0].item()
            start = row - 2 * n
            assert torch.allclose(batch.observations["policy"][i], obs_all[start])
            assert torch.allclose(batch.hidden[:, i], buf.hidden[start, 0])
            assert torch.allclose(batch.bootstrap_hidden[:, i], buf.hidden[row, 0])
            with torch.no_grad():
                h_next = alg.actor.step_state(
                    batch.bootstrap_observations[i : i + 1], batch.bootstrap_hidden[:, i : i + 1]
                )
            if row + n < len(buf):
                assert torch.allclose(h_next[:, 0], buf.hidden[row + n, 0], atol=1e-6)

    def test_episode_start_windows_begin_at_resets(self) -> None:
        """Episode-start windows open on a step whose predecessor was a done, so they start from a zero state."""
        torch.manual_seed(0)
        alg, env = _build(done_prob=0.2)
        _collect(alg, env, 24)
        batch = alg.replay_buffer.sample_sequences(alg.seq_len, 0, alg.device, episode_start=True)
        assert batch is not None
        assert torch.all(batch.init_hidden == 0)

    def test_aux_head_trains_on_privileged_target(self) -> None:
        """With an aux group the actor grows a head sized to it and the update reports its loss."""
        torch.manual_seed(0)
        alg, env = _build(aux_obs_group="target")
        assert alg.actor.aux_head is not None and alg.actor.aux_head.out_features == 3
        _collect(alg, env, 12)
        _, diag = alg.update()
        assert "Actor/aux_loss" in diag and diag["Actor/aux_loss"] > 0
        assert torch.isfinite(torch.tensor(diag["Actor/aux_abs_err"]))

    def test_no_aux_head_by_default(self) -> None:
        """Without an aux group the actor has no aux head and the update reports no aux loss."""
        alg, env = _build()
        assert alg.actor.aux_head is None
        _collect(alg, env, 12)
        _, diag = alg.update()
        assert "Actor/aux_loss" not in diag


def _build_state_only(
    done_prob: float = 0.1, obs_normalization: bool = True, aux_obs_group: str | None = None
) -> tuple[SAC, _DummyImageVecEnv]:
    """Build SAC with an RNNModel actor that reads only the 1D policy group."""
    env = _DummyImageVecEnv(done_prob=done_prob)
    cfg = _make_cfg(obs_normalization, aux_obs_group)
    cfg["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}
    cfg["actor"]["class_name"] = "RNNModel"
    del cfg["actor"]["cnn_cfg"]
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
    return alg, env


class TestStateOnlyRecurrentSAC:
    """A plain RNNModel actor over 1D observations, as for a state-based teacher with memory."""

    def test_builds_a_recurrent_actor(self) -> None:
        """The head reads the RNN state alone."""
        alg, _ = _build_state_only()
        assert alg.recurrent and alg.actor._get_latent_dim() == 16

    def test_stored_state_is_self_consistent(self) -> None:
        """Stored states chain across steps and zero on done, as with an image group."""
        torch.manual_seed(0)
        alg, env = _build_state_only(done_prob=0.5, obs_normalization=False)
        dones = _collect(alg, env, 6)
        buf, n = alg.replay_buffer, env.num_envs
        for t in range(5):
            row_t, row_next = t * n, (t + 1) * n
            h_t = buf.hidden[row_t : row_t + n, 0].permute(1, 0, 2)
            with torch.no_grad():
                _, h_after = alg.actor.encode_sequence(buf.observations[row_t : row_t + n].unsqueeze(0), h_t)
            h_stored_next = buf.hidden[row_next : row_next + n, 0].permute(1, 0, 2)
            assert torch.allclose(h_after[:, ~dones[t]], h_stored_next[:, ~dones[t]], atol=1e-6)
            if dones[t].any():
                assert torch.all(h_stored_next[:, dones[t]] == 0)

    def test_update_runs_and_is_finite(self) -> None:
        """The windowed update runs every loss on a state-only recurrent actor."""
        torch.manual_seed(0)
        alg, env = _build_state_only()
        _collect(alg, env, 12)
        losses, _ = alg.update()
        for key in ("critic_1", "critic_2", "actor", "alpha"):
            assert torch.isfinite(torch.tensor(losses[key])), key

    def test_aux_head_estimates_a_privileged_target(self) -> None:
        """An aux group gives the RNNModel actor a head on its recurrent state and the update trains it."""
        torch.manual_seed(0)
        alg, env = _build_state_only(aux_obs_group="target")
        assert alg.actor.aux_head is not None and alg.actor.aux_head.in_features == 16
        assert alg.actor.aux_head.out_features == 3
        _collect(alg, env, 12)
        _, diag = alg.update()
        assert "Actor/aux_loss" in diag and diag["Actor/aux_loss"] > 0


def _build_with_estimator(image_actor: bool) -> tuple[SAC, _DummyImageVecEnv]:
    """Build SAC with a separate supervised estimator of the ``target`` group."""
    env = _DummyImageVecEnv()
    cfg = _make_cfg()
    if not image_actor:
        cfg["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}
        cfg["actor"] = {k: v for k, v in cfg["actor"].items() if k != "cnn_cfg"} | {"class_name": "RNNModel"}
    cfg["algorithm"] |= {"estimator_obs_group": "target", "estimator_view_group": "view", "estimator_hidden_dim": 8}
    return SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu"), env


class TestSequenceEstimator:
    """The separate estimator trains on replay windows, logs split errors, and checkpoints."""

    @pytest.mark.parametrize("image_actor", [False, True])
    def test_estimator_trains_and_logs_split_error(self, image_actor: bool) -> None:
        """It reads only the actor's vector groups, steps its own optimizer, and logs masked and visible error."""
        torch.manual_seed(0)
        alg, env = _build_with_estimator(image_actor)
        assert alg.estimator is not None and alg.estimator.readout.out_features == 3
        assert alg.estimator.norm.normalized_shape == (OBS_DIM,)
        assert getattr(alg.actor, "aux_head", None) is None
        before = [p.detach().clone() for p in alg.estimator.parameters()]
        _collect(alg, env, 12)
        _, diag = alg.update()
        for key in ("Estimator/loss", "Estimator/abs_err", "Estimator/abs_err_mean_guess", "Estimator/masked_frac"):
            assert key in diag and torch.isfinite(torch.tensor(diag[key])), key
        assert "Estimator/abs_err_masked" in diag and "Estimator/abs_err_visible" in diag
        assert any(not torch.equal(b, p) for b, p in zip(before, alg.estimator.parameters(), strict=True))

    def test_estimator_round_trips_through_a_checkpoint(self) -> None:
        """Its weights save with the algorithm and load back."""
        torch.manual_seed(0)
        alg, env = _build_with_estimator(image_actor=False)
        _collect(alg, env, 12)
        alg.update()
        saved = alg.save()
        assert "estimator_state_dict" in saved
        other, _ = _build_with_estimator(image_actor=False)
        other.load(saved)
        for a, b in zip(alg.estimator.parameters(), other.estimator.parameters(), strict=True):
            assert torch.equal(a, b)


class _TimeoutImageVecEnv(_DummyImageVecEnv):
    """Half of the dones are timeouts, with a true next observation the buffer stores in place of the reset one."""

    def step(self, actions: torch.Tensor) -> tuple:
        next_obs, rewards, dones, extras = super().step(actions)
        extras = {"time_outs": dones & (torch.rand(self.num_envs) < 0.5), "time_outs_obs": self._obs()}
        return next_obs, rewards, dones, extras


def _old_reads(buffer: ReplayBuffer) -> None:
    """Make ``buffer`` read every group of every batch, dequantizing out of place, as before reads were trimmed."""
    import types

    sample_mini_batch, sample_sequences = buffer.sample_mini_batch, buffer.sample_sequences

    def _read(
        self: ReplayBuffer, observations: TensorDict, idx: torch.Tensor, device: str | None, groups: object = None
    ) -> TensorDict:
        out = observations[idx].to(device)
        for key, dtype in self._obs_dtypes.items():
            if key in self._quantize:
                low, high = self._quantize[key]
                out[key] = out[key].to(dtype) * ((high - low) / 255.0) + low
            elif out[key].dtype != dtype:
                out[key] = out[key].to(dtype)
        return out

    buffer._read = types.MethodType(_read, buffer)
    buffer.sample_mini_batch = lambda device=None, obs_groups=None: sample_mini_batch(device)
    buffer.sample_sequences = lambda *a, next_observations=True, **k: sample_sequences(*a, **k)


class TestTrimmedReads:
    """Skipping the image groups an update never reads leaves every loss and parameter bit-identical."""

    @pytest.mark.parametrize("n_steps", [1, 5])
    @pytest.mark.parametrize("critic_sees_hidden", [False, True])
    def test_update_matches_full_reads(self, n_steps: int, critic_sees_hidden: bool) -> None:
        """Same seed and buffer: the trimmed and the full-read update agree exactly, over timeouts and resets."""
        import copy

        torch.manual_seed(0)
        env = _TimeoutImageVecEnv(done_prob=0.3)
        cfg = _make_cfg(n_steps=n_steps)
        cfg["algorithm"] |= {"replay_obs_quantize": {"image": [0.0, 1.0]}, "critic_sees_hidden": critic_sees_hidden}
        cfg["algorithm"] |= {"seq_len": 4, "burn_in": 2, "mini_batch_size": 16, "num_mini_batches": 3}
        alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
        dones = _collect(alg, env, 16)
        assert torch.stack(dones).any()
        old = copy.deepcopy(alg)
        _old_reads(old.replay_buffer)
        torch.manual_seed(1)
        losses, diag = alg.update()
        torch.manual_seed(1)
        old_losses, old_diag = old.update()
        assert losses == old_losses
        assert diag == old_diag
        for a, b in zip(alg.save().values(), old.save().values(), strict=True):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
        for a, b in zip(alg.actor.parameters(), old.actor.parameters(), strict=True):
            assert torch.equal(a, b)
        for a, b in zip(alg.critic_parameters, old.critic_parameters, strict=True):
            assert torch.equal(a, b)
        assert torch.equal(alg.log_alpha, old.log_alpha)


def _build_on_windows(
    n_steps: int, seq_len: int, burn_in: int, done_prob: float = 0.3
) -> tuple[SAC, _DummyImageVecEnv]:
    env = _TimeoutImageVecEnv(done_prob=done_prob)
    cfg = _make_cfg(obs_normalization=False, n_steps=n_steps)
    cfg["algorithm"] |= {"critic_on_actor_windows": True, "seq_len": seq_len, "burn_in": burn_in}
    cfg["algorithm"] |= {"replay_obs_quantize": {"image": [0.0, 1.0]}, "mini_batch_size": 64, "num_mini_batches": 2}
    return SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu"), env


class TestCriticOnActorWindows:
    """The critics train on the actor's loss steps, with n-step targets bootstrapped through its window unroll."""

    def test_nstep_returns_stop_at_the_first_episode_end(self) -> None:
        """A return sums up to and including its first done, inside the loss steps or in the extension."""
        from robot_rl.algorithms.sac import _window_nstep_returns

        rewards = torch.tensor([[1.0], [2.0], [4.0], [8.0], [16.0]])  # seq_len 2, n 4: 5 steps
        dones = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]])
        returns, horizon, final = _window_nstep_returns(rewards, dones, 2, 4, 0.5)
        torch.testing.assert_close(returns[:, 0], torch.tensor([1.0 + 1.0 + 1.0, 2.0 + 2.0]))
        assert horizon[:, 0].tolist() == [3, 2] and final[:, 0].tolist() == [2, 2]
        returns, horizon, final = _window_nstep_returns(rewards, torch.zeros_like(dones), 2, 4, 0.5)
        torch.testing.assert_close(returns[:, 0], torch.tensor([1.0 + 1.0 + 1.0 + 1.0, 2.0 + 2.0 + 2.0 + 2.0]))
        assert horizon[:, 0].tolist() == [4, 4] and final[:, 0].tolist() == [3, 4]

    @pytest.mark.parametrize("n_steps", [1, 3, 5])
    def test_one_step_windows_give_the_flat_nstep_targets(self, n_steps: int) -> None:
        """A loss step with its n-1 step extension gets the flat sampler's n-step target, over dones and timeouts.

        Weights and normalization are fixed, so the window's replayed state equals the stored one the flat
        target steps from; the action head is made deterministic so both sample the same bootstrap action.
        """
        torch.manual_seed(0)
        alg, env = _build_on_windows(n_steps, seq_len=1, burn_in=0)
        _collect(alg, env, 24)
        actor = alg.actor

        def mean_action(latent: torch.Tensor, std_clip: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            out = actor.mlp(latent)
            return torch.tanh(out), out.sum(dim=-1)

        actor.act_and_log_prob_from_latent = mean_action
        buf = alg.replay_buffer
        batch = buf.sample_sequences(1, 0, "cpu", num_windows=256, next_observations=["critic"], extend=n_steps - 1)
        assert batch.observations.batch_size[0] == n_steps
        # episode ends that cut a return short: terminations and timeouts, in the loss step and in the extension
        ends = batch.dones[: max(n_steps - 1, 1)] > 0
        timeouts = ends & (batch.next_terminated[: ends.shape[0]].squeeze(-1) == 0)
        assert (ends & ~timeouts).any() and timeouts.any()
        if n_steps > 2:
            assert ends[1:].any()
        with torch.no_grad():
            latent, state, states = actor.encode_sequence_with_states(
                batch.observations[:1], batch.init_hidden, batch.resets[:1]
            )
            _, _, target, _ = alg._window_critic_target(batch, latent, state, states)
            start = batch.indices[0]
            flat = buf._nstep_batch(start // buf.num_envs, start % buf.num_envs, "cpu") if n_steps > 1 else None
            if flat is None:
                flat = ReplayBuffer.Batch(
                    buf._read(buf.observations, start, "cpu"),
                    buf._read(buf.next_observations, start, "cpu"),
                    buf.actions[start],
                    buf.rewards[start],
                    buf.gammas[start],
                    buf.context[start],
                    buf.next_terminated[start],
                    hidden=buf._gather_hidden(start, "cpu"),
                )
            flat_target, _ = alg._flat_critic_target(flat)
        torch.testing.assert_close(target, flat_target, rtol=1e-5, atol=1e-5)

    def test_update_trains_on_the_actor_windows(self) -> None:
        """The update runs, keeps today's window count, and moves both critics and the actor."""
        torch.manual_seed(0)
        alg, env = _build_on_windows(n_steps=3, seq_len=4, burn_in=2)
        _collect(alg, env, 16)
        batch = alg.replay_buffer.sample_sequences(4, 2, "cpu", extend=2)
        assert batch.observations.batch_size == (8, 64 // 6)
        critic_before = [p.detach().clone() for p in alg.critic_parameters]
        actor_before = [p.detach().clone() for p in alg.actor.parameters()]
        flat_groups = []
        sample_mini_batch = alg.replay_buffer.sample_mini_batch

        def record(*args: object, obs_groups: object = None) -> ReplayBuffer.Batch:
            flat_groups.append(obs_groups)
            return sample_mini_batch(*args, obs_groups=obs_groups)

        alg.replay_buffer.sample_mini_batch = record
        losses, diag = alg.update()
        assert flat_groups == [()]  # only the end-of-update reward check; no flat critic batch
        for key in ("critic_1", "critic_2", "actor", "alpha"):
            assert torch.isfinite(torch.tensor(losses[key])), key
        assert "Critic/td_abs" in diag and "Critic/q_pi_gap" in diag
        assert any(not torch.equal(a, p) for a, p in zip(critic_before, alg.critic_parameters, strict=True))
        assert any(not torch.equal(a, p) for a, p in zip(actor_before, alg.actor.parameters(), strict=True))

    def test_excludes_critic_sees_hidden(self) -> None:
        """The critic input stays observation and action, so the latent-reading critic cannot be combined."""
        env = _TimeoutImageVecEnv()
        cfg = _make_cfg()
        cfg["algorithm"] |= {"critic_on_actor_windows": True, "critic_sees_hidden": True}
        with pytest.raises(ValueError, match="critic_on_actor_windows"):
            SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
