# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the CReF actor inside recurrent SAC (dummy VecEnv with an image group, no Isaac Sim)."""

import torch
from tensordict import TensorDict

from robot_rl.algorithms import SAC
from tests.algorithms.test_sac_recurrent import NUM_ENVS, _collect, _DummyImageVecEnv, _make_cfg


def _build(
    aux_obs_group: str | None = None, attn_target_group: str | None = None, aux_readout: bool = False
) -> tuple[SAC, _DummyImageVecEnv]:
    env = _DummyImageVecEnv()
    cfg = _make_cfg(aux_obs_group=aux_obs_group)
    cfg["algorithm"]["attn_target_group"] = attn_target_group
    cfg["actor"] = {
        "class_name": "CrefModel",
        "hidden_dims": [32],
        "obs_normalization": True,
        "token_dim": 16,
        "num_heads": 2,
        "rnn_type": "gru",
        "rnn_hidden_dim": 16,
        "rnn_num_layers": 1,
        "cnn_cfg": {"output_channels": [4, 4], "kernel_size": 3, "stride": 2, "activation": "elu"},
        "distribution_cfg": {"class_name": "SquashedTanhGaussianDistribution", "init_noise_std": 1.0},
        "aux_readout": aux_readout,
    }
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device="cpu")
    return alg, env


class TestCrefModel:
    """Tokens, attention, fusion and the highway gate behave inside the recurrent SAC pipeline."""

    def test_head_input_width_and_tokens(self) -> None:
        """The head reads the fused width and attention spans one token per feature-map cell."""
        alg, env = _build()
        obs = env.get_observations()
        alg.actor.reset()
        with torch.no_grad():
            latent = alg.actor.get_latent(obs)
        assert latent.shape == (NUM_ENVS, 32)
        assert alg.actor._last_attn.shape[0] == NUM_ENVS and alg.actor._last_attn.shape[1] > 1
        assert torch.allclose(alg.actor._last_attn.sum(dim=-1), torch.ones(NUM_ENVS), atol=1e-5)
        entropy = alg.actor.attention_entropy()
        assert 0.0 < float(entropy) <= 1.0

    def test_sequence_matches_stepwise_rollout(self) -> None:
        """Encoding a window equals stepping the rollout RNN through it (gate and attention included)."""
        torch.manual_seed(0)
        alg, env = _build()
        obs_seq = [env.get_observations() for _ in range(5)]
        alg.actor.reset()
        with torch.no_grad():
            stepwise = [alg.actor.get_latent(obs) for obs in obs_seq]
            latent, _ = alg.actor.encode_sequence(TensorDict.stack(obs_seq, dim=0), None)
        assert torch.allclose(latent, torch.stack(stepwise), atol=1e-6)

    def test_update_runs_with_aux_and_attention_metrics(self) -> None:
        """The recurrent update trains the CReF actor and reports attention entropy and the aux loss."""
        torch.manual_seed(0)
        alg, env = _build(aux_obs_group="target")
        _collect(alg, env, 12)
        losses, diag = alg.update()
        for key in ("critic_1", "actor", "alpha"):
            assert torch.isfinite(torch.tensor(losses[key])), key
        assert 0.0 < diag["Actor/attn_entropy"] <= 1.0
        assert diag["Actor/aux_loss"] > 0 and diag["Actor/grad_norm_rnn"] > 0

    def test_attention_target_log_prob_indexes_the_grid(self) -> None:
        """The target cell's log attention mass matches a direct lookup on the (rows, cols) token grid."""
        alg, env = _build()
        obs = env.get_observations()
        alg.actor.reset()
        with torch.no_grad():
            alg.actor.get_latent(obs)
            rows, cols = alg.actor.token_grids[0]
            assert rows * cols == alg.actor._last_attn.shape[1]
            uv = torch.tensor([[0.99, 0.99]] * NUM_ENVS)
            log_p = alg.actor.attention_target_log_prob(uv)
        assert torch.allclose(log_p.exp(), alg.actor._last_attn[:, -1], atol=1e-6)

    def test_attention_supervision_raises_target_mass(self) -> None:
        """Training with an attention target reports its loss, and the attended mass on the target grows."""
        torch.manual_seed(0)
        alg, env = _build(attn_target_group="pixel")
        _collect(alg, env, 12)
        first = alg.update()[1]["Actor/attn_target_prob"]
        for _ in range(15):
            last = alg.update()[1]["Actor/attn_target_prob"]
        assert first > 0.0 and last > first

    def test_perception_warmup_freezes_the_rl_head(self) -> None:
        """During warm-up the head's output distribution and alpha stay put while the encoder still trains."""
        torch.manual_seed(0)
        alg, env = _build(aux_obs_group="target", attn_target_group="pixel")
        alg.perception_warmup_updates = 10**6
        _collect(alg, env, 12)
        head_before = [p.detach().clone() for p in alg.actor.mlp.parameters()]
        attn_before = [p.detach().clone() for p in alg.actor.attn.parameters()]
        alpha_before = alg.alpha
        _, diag = alg.update()
        assert diag["Actor/warming_up"] == 1.0 and alg.alpha == alpha_before
        assert all(torch.equal(a, b) for a, b in zip(head_before, alg.actor.mlp.parameters(), strict=True))
        assert any(not torch.equal(a, b) for a, b in zip(attn_before, alg.actor.attn.parameters(), strict=True))

    def test_aux_readout_appends_prediction_to_head_input(self) -> None:
        """With aux_readout the head input ends with the aux prediction, and the update still trains."""
        torch.manual_seed(0)
        alg, env = _build(aux_obs_group="target", aux_readout=True)
        obs = env.get_observations()
        alg.actor.reset()
        with torch.no_grad():
            latent = alg.actor.get_latent(obs)
        assert latent.shape[-1] == 32 + 3
        assert torch.allclose(latent[:, -3:], alg.actor.aux_prediction(latent), atol=1e-6)
        _collect(alg, env, 12)
        losses, diag = alg.update()
        assert torch.isfinite(torch.tensor(losses["actor"])) and diag["Actor/aux_loss"] > 0
