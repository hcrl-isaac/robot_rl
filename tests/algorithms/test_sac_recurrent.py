# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for SAC with a recurrent CNN actor (dummy VecEnv with an image group, no Isaac Sim)."""

import torch
from tensordict import TensorDict

from robot_rl.algorithms import SAC

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


def _make_cfg(obs_normalization: bool = True, aux_obs_group: str | None = None) -> dict:
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
            "n_steps": 1,
            "seq_len": 4,
            "burn_in": 2,
            "aux_obs_group": aux_obs_group,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def _build(
    done_prob: float = 0.1, obs_normalization: bool = True, aux_obs_group: str | None = None
) -> tuple[SAC, _DummyImageVecEnv]:
    env = _DummyImageVecEnv(done_prob=done_prob)
    cfg = _make_cfg(obs_normalization, aux_obs_group)
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
