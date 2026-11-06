"""CPU tests for FB-CPR with an imagined actor objective (the world-model AC's toy env, no Isaac Sim)."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

import pytest

from robot_rl.algorithms import FbCpr, WorldModelFbCpr
from robot_rl.env import ObsTermLayout, WorldModelSpec
from tests.algorithms.test_world_model_ac import _TrackingEnv

Z_DIM, BATCH = 8, 64


class _ExpertBuffer:
    """Stands in for the motion dataset: random expert states."""

    def __init__(self, motion_path: str, groups: list[str], device: str | None) -> None:
        self.groups = groups

    def sample(self, n: int, device: str, seq_length: int = 1) -> tuple[TensorDict, TensorDict]:
        def draw() -> TensorDict:
            return TensorDict({g: torch.randn(n, 4) for g in self.groups}, batch_size=[n])

        return draw(), draw()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def _env() -> _TrackingEnv:
    # FB-CPR keeps every env's episode in step, so nothing terminates early; it conditions on z, not a goal
    env = _TrackingEnv(num_envs=16, max_error=float("inf"))
    env.world_model_spec = WorldModelSpec(
        state_group="state",
        obs_group="obs_bar",
        context_group="context",
        action_terms=["last_action"],
        layouts={
            "policy": [ObsTermLayout("pos", 2, 2), ObsTermLayout("vel", 2, 2), ObsTermLayout("last_action", 2, 2)],
            "obs_bar": [ObsTermLayout("pos", 2), ObsTermLayout("vel", 2)],
        },
    )
    return env


def _cfg(horizon: int, actor_start_update: int = 0) -> dict:
    fuse = {"class_name": "ResidualFuseModel", "embedding_dims": [32], "hidden_dims": [32, 32], "num_parallel": 2}
    mlp = {"class_name": "MLPModel", "hidden_dims": [32]}
    return {
        "actor": {
            "class_name": "ResidualFuseModel",
            "embedding_dims": [32],
            "hidden_dims": [32, 32],
            "distribution_cfg": {"class_name": "TruncatedGaussianDistribution", "init_std": 0.25},
        },
        "algorithm": {
            "class_name": "WorldModelFbCpr",
            "z_dim": Z_DIM,
            "z_buffer_capacity": 128,
            "motion_path": "",
            "expert_sequence_length": 2,
            "steps_per_z_update": 10,
            "storage_scale": 2,
            "batch_size": BATCH,
            "compile_mode": None,
            "imagination_compile_mode": None,
            "forward_map": dict(fuse),
            "backward_map": {**mlp, "last_activation": "ball_norm"},
            "disc_critic": dict(fuse),
            "aux_critic": dict(fuse),
            "discriminator": {**mlp, "last_activation": "sigmoid"},
            "world_model": {"hidden_dims": [64, 64], "head_hidden_dims": [32]},
            "wm_window": 4,
            "wm_batch_size": 64,
            "wm_mini_batches": 2,
            "imagination_horizon": horizon,
            "noise_bank_size": 256,
            "actor_start_update": actor_start_update,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["state", "context"],
            "backward": ["state"],
            "discriminator": ["state"],
            "expert": ["state"],
        },
        "storage_device": "cpu",
        "multi_gpu": None,
        "clip_actions": 5.0,
        "num_seed_steps_per_env": 0,
    }


@pytest.fixture(autouse=True)
def _stub_motions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("robot_rl.algorithms.fb_cpr.TrajectoryBuffer", _ExpertBuffer)


def _build(horizon: int, actor_start_update: int = 0) -> tuple[WorldModelFbCpr, _TrackingEnv]:
    env = _env()
    alg = WorldModelFbCpr.construct_algorithm(env.get_observations(), env, _cfg(horizon, actor_start_update), "cpu")
    obs = env.get_observations()
    for _ in range(30):
        with torch.inference_mode():
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    return alg, env


@pytest.mark.parametrize("horizon", [0, 4])
def test_update_trains_every_part_and_round_trips(horizon: int) -> None:
    """An update logs finite FB, critic, world-model and imagined-actor losses, and a checkpoint restores."""
    alg, env = _build(horizon)
    before = copy.deepcopy(alg.actor.state_dict())
    loss_dict, extras = alg.update()
    for key in ("Forward_Backward_Loss/fb_loss", "Actor_Loss/total_loss", "world_model/state"):
        assert key in loss_dict and torch.isfinite(loss_dict[key]).all(), (key, loss_dict)
    assert int(loss_dict["Actor_Loss/horizon"]) == horizon
    assert ("imagined_fb_reward" in extras) == (horizon > 0)
    assert any((before[k] != v).any() for k, v in alg.actor.state_dict().items())
    restored = WorldModelFbCpr.construct_algorithm(env.get_observations(), env, _cfg(horizon), "cpu")
    restored.load(copy.deepcopy(alg.save()), None, strict=True)
    for k, v in alg.world_model.state_dict().items():
        torch.testing.assert_close(restored.world_model.state_dict()[k], v)


def test_actor_imagines_from_the_start_update_at_its_interval() -> None:
    """The actor takes FB-CPR's one-step loss before ``actor_start_update`` and between imagined updates."""
    alg, _ = _build(horizon=4, actor_start_update=2)
    alg.imagined_actor_interval, alg.num_imagined = 2, 16
    horizons = [int(alg.update()[0]["Actor_Loss/horizon"]) for _ in range(5)]
    assert horizons == [0, 0, 4, 0, 4]


def test_horizon_zero_is_fb_cpr_actor_loss() -> None:
    """With no imagined steps the actor loss equals FB-CPR's, on the same batch, latents and action noise."""
    alg, _ = _build(horizon=0)
    win = alg.learner.replay_buffer.sample_transitions(BATCH)
    batch = alg._as_batch(win)
    alg.obs_normalizer(batch.observations)
    with torch.no_grad():
        batch.observations = alg._normalize(batch.observations)
        batch.next_observations = alg._normalize(batch.next_observations)
    z = alg.project_z(torch.randn(BATCH, Z_DIM))
    batch.context = z
    reference = copy.deepcopy(alg)
    torch.manual_seed(0)
    imagined, _ = alg._update_imagined_actor(win.obs[0], z, batch.next_observations, horizon=0)
    torch.manual_seed(0)
    one_step, _ = FbCpr._update_actor(reference, batch)
    torch.testing.assert_close(imagined["Actor_Loss/total_loss"], one_step["Actor_Loss/total_loss"])


def _model_cfg_alg(**algorithm: object) -> WorldModelFbCpr:
    env = _env()
    cfg = _cfg(horizon=4)
    cfg["algorithm"].update(algorithm)
    alg = WorldModelFbCpr.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    obs = env.get_observations()
    for _ in range(30):
        with torch.inference_mode():
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    return alg


def _normalized_batch(alg: WorldModelFbCpr) -> tuple:
    win = alg.learner.replay_buffer.sample_transitions(BATCH)
    batch = alg._as_batch(win)
    batch.observations = alg._normalize(batch.observations)
    batch.next_observations = alg._normalize(batch.next_observations)
    batch.context = alg.project_z(torch.randn(BATCH, Z_DIM))
    batch.rewards = alg.aux_reward_normalizer(batch.rewards)  # as update() does before the critics
    return win, batch


def test_critic_hooks_at_their_defaults_are_fb_cprs_updates() -> None:
    """The real next state, the real discount and zero extra credit or reward reproduce each critic loss exactly."""
    alg = _model_cfg_alg()
    _, batch = _normalized_batch(alg)
    zeros = torch.zeros(BATCH, 1)
    calls = [
        ("_update_forward_backward", {"credit_backward": torch.zeros(BATCH, Z_DIM)}),
        ("_update_disc_critic", {"extra_rewards": zeros}),
        ("_update_aux_critic", {"extra_rewards": zeros}),
    ]
    for name, extra in calls:
        plain, hooked = copy.deepcopy(alg), copy.deepcopy(alg)
        explicit = {"bootstrap_obs": batch.next_observations, "bootstrap_gammas": batch.gammas, **extra}
        torch.manual_seed(0)
        a, _ = getattr(plain, name)(copy.deepcopy(batch))
        torch.manual_seed(0)
        b, _ = getattr(hooked, name)(copy.deepcopy(batch), **explicit)
        for key in a:
            torch.testing.assert_close(a[key], b[key], msg=f"{name} {key}")


@pytest.mark.parametrize("mage_action", ["policy", "buffer"])
def test_mve_and_mage_update_and_log(mage_action: str) -> None:
    """With k-step targets and the action-gradient loss on, an update trains every critic and logs both."""
    alg = _model_cfg_alg(
        mve_k=3, mage_beta=0.1, mage_q_beta=1.0, mage_batch_size=16, mage_action=mage_action, value_loss_coef=0.1
    )
    loss_dict, extras = alg.update()
    for key in ("Forward_Backward_Loss/fb_loss", "Critic_Loss/discriminator_critic_loss", "Actor_Loss/total_loss"):
        assert torch.isfinite(loss_dict[key]).all(), key
    for key in ("mve_fb_credit_norm", "mve_disc_reward", "mage_forward_action_grad", "mage_aux_action_grad"):
        assert key in extras and torch.isfinite(extras[key]), (key, extras)


@pytest.mark.parametrize("directions", [None, 2])
@pytest.mark.parametrize("critic", ["forward", "disc", "aux"])
def test_mage_trains_only_its_critic(critic: str, directions: int | None) -> None:
    """The action-gradient loss (exact or finite-difference) reaches the trained critic's parameters only."""
    alg = _model_cfg_alg(mage_beta=1.0, mage_directions=directions)
    win, batch = _normalized_batch(alg)
    for m in (alg.actor, alg.backward_map, alg.discriminator, alg.world_model, alg.forward_map, alg.disc_critic):
        m.zero_grad(set_to_none=True)
    alg.aux_critic.zero_grad(set_to_none=True)
    alg._mage_backward(critic, win, batch)
    trained = {"forward": alg.forward_map, "disc": alg.disc_critic, "aux": alg.aux_critic}[critic]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained.parameters())
    others = [alg.actor, alg.backward_map, alg.discriminator, alg.world_model]
    others += [m for m in (alg.forward_map, alg.disc_critic, alg.aux_critic) if m is not trained]
    for m in others:
        assert all(p.grad is None for p in m.parameters()), type(m).__name__
    for target in (alg.target_forward_map, alg.target_disc_critic, alg.target_aux_critic):
        assert all(p.grad is None for p in target.target.parameters())


def test_implicit_reward_weights_survive_a_rank_deficient_backward_map() -> None:
    """Cov(B) over a batch whose B spans fewer than z_dim directions still yields finite reward weights."""
    alg = _model_cfg_alg()
    _, batch = _normalized_batch(alg)
    direction = torch.nn.functional.normalize(torch.randn(Z_DIM), dim=0)
    alg.backward_map.forward = lambda obs: direction.expand(obs.batch_size[0], -1)  # rank one
    weights = alg._implicit_reward_weights(batch.next_observations, batch.context)
    assert torch.isfinite(weights).all()


def test_mage_gradient_is_clipped_and_a_non_finite_one_dropped() -> None:
    """The MAGE parameter gradient written into the critic respects the norm clip, and a NaN one leaves it zero."""
    alg = _model_cfg_alg(mage_beta=1e6, mage_max_grad_norm=0.5)
    win, batch = _normalized_batch(alg)
    alg._mage_backward("forward", win, batch)
    norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in alg.forward_map.parameters() if p.grad is not None])
    )
    assert float(norm) <= 0.5 + 1e-4
    alg.mage_beta = float("nan")
    out = alg._mage_backward("forward", win, batch)
    assert float(out["mage_forward_dropped"]) == 1.0
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in alg.forward_map.parameters())


def test_finite_difference_mage_matches_the_exact_slope() -> None:
    """With many directions and a small step, the finite-difference |∂δ/∂a| agrees with double backprop."""
    alg = _model_cfg_alg(mage_beta=1.0)
    win, batch = _normalized_batch(alg)
    torch.manual_seed(0)
    exact = alg._mage_backward("disc", win, batch)["mage_disc_action_grad"]
    alg.mage_directions, alg.mage_fd_eps = 256, 1e-3
    torch.manual_seed(0)
    estimate = alg._mage_backward("disc", win, batch)["mage_disc_action_grad"]
    torch.testing.assert_close(estimate, exact, rtol=0.15, atol=1e-4)


def test_mve_can_leave_the_forward_backward_loss_one_step() -> None:
    """With F excluded from mve_critics, the FB update gets no expansion while the scalar critics do."""
    alg = _model_cfg_alg(mve_k=3, mve_critics=("disc", "aux"), value_loss_coef=0.0)
    win, batch = _normalized_batch(alg)
    fb_kwargs, disc_kwargs, aux_kwargs, _ = alg._model_critic_inputs(win, batch)
    assert fb_kwargs == {} and "extra_rewards" in disc_kwargs and "extra_rewards" in aux_kwargs


def test_value_regularizer_takes_the_expanded_target() -> None:
    """The Fz target adds the credited states' implicit rewards and bootstraps with the expanded discount."""
    alg = _model_cfg_alg(value_loss_coef=0.1)
    _, batch = _normalized_batch(alg)
    credit = torch.randn(BATCH, Z_DIM)
    zero_gammas = torch.zeros(BATCH, 1)
    with torch.no_grad():
        Fs = alg.forward_map(batch.observations, batch.context, batch.actions)
        B = alg.backward_map(batch.next_observations)
        cov = B.T @ B / B.shape[0]
        target = (torch.linalg.solve(cov, B + credit, left=False) * batch.context).sum(dim=-1)
        Qs = (Fs * batch.context).sum(dim=-1)
        expected = 0.1 * 0.5 * Fs.shape[0] * torch.nn.functional.mse_loss(Qs, target.expand_as(Qs))
    losses, _ = alg._update_forward_backward(
        batch, bootstrap_obs=batch.next_observations, bootstrap_gammas=zero_gammas, credit_backward=credit
    )
    torch.testing.assert_close(losses["Forward_Backward_Loss/value_reg_loss"], expected)


@pytest.mark.parametrize("ridge", [0.0, 0.5])
def test_credit_ridge_shrinks_the_weak_directions_of_real_backward_covariance(ridge: float) -> None:
    """Each covariance eigendirection keeps eig / (eig + ridge * mean eig) of the credit; 0 leaves it untouched."""
    alg = _model_cfg_alg(mve_k=3, mve_credit_ridge=ridge)
    _, batch = _normalized_batch(alg)
    with torch.no_grad():
        own = alg.backward_map(batch.next_observations)
        _, vec = torch.linalg.eigh(own.T @ own / own.shape[0])
    credit = torch.stack([vec[:, -1], vec[:, 0]])
    batch.next_observations = batch.next_observations[:2]
    shrunk, extras = alg._shrink_credit(credit, batch)
    if ridge == 0:
        assert shrunk is credit
    else:
        eig2, vec2 = torch.linalg.eigh((lambda b: b.T @ b / b.shape[0])(own[:2]))
        gain = eig2.clamp_min(0) / (eig2.clamp_min(0) + ridge * eig2.clamp_min(0).mean())
        expected = ((credit @ vec2) * gain) @ vec2.T
        torch.testing.assert_close(shrunk, expected, atol=1e-5, rtol=1e-4)
        assert shrunk.norm(dim=-1).le(credit.norm(dim=-1) + 1e-6).all()
    assert {"mve_cov_eig_p1", "mve_credit_weak_share_0.01", "mve_own_inv_cov_norm"} <= set(extras)


@pytest.mark.parametrize("credit_grad", [False, True])
def test_credit_grad_trains_backward_on_imagined_states_but_not_the_model(credit_grad: bool) -> None:
    """With mve_credit_grad the credit carries B's gradient (none reaches the world model); off, it is detached."""
    alg = _model_cfg_alg(mve_k=3, mve_credit_grad=credit_grad, value_loss_coef=0.0)
    alg.update_step = alg.critic_model_start_update
    win, batch = _normalized_batch(alg)
    fb_kwargs, _, _, _ = alg._model_critic_inputs(win, batch)
    credit = fb_kwargs["credit_backward"]
    assert credit.requires_grad == credit_grad
    if credit_grad:
        alg.backward_map.zero_grad()
        credit.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.backward_map.parameters())
        assert all(p.grad is None for p in alg.world_model.parameters())


def test_real_states_as_negatives_leave_the_fb_loss_unchanged() -> None:
    """A mixture of the real off-diagonal penalty with itself is the one-step FB loss."""
    alg = _model_cfg_alg()
    _, batch = _normalized_batch(alg)
    plain, mixed = copy.deepcopy(alg), copy.deepcopy(alg)
    torch.manual_seed(0)
    a, _ = plain._update_forward_backward(copy.deepcopy(batch))
    torch.manual_seed(0)
    b, _ = mixed._update_forward_backward(
        copy.deepcopy(batch),
        negatives=batch.next_observations,
        negative_mask=1 - torch.eye(BATCH),
        negative_weight=0.5,
    )
    torch.testing.assert_close(a["Forward_Backward_Loss/fb_loss"], b["Forward_Backward_Loss/fb_loss"])


def test_imagined_negatives_skip_each_rows_own_rollout() -> None:
    """Each row is penalized against every other live row's imagined states, never its own."""
    alg = _model_cfg_alg(mve_k=3, mve_credit_grad=True, mve_imagined_negative_weight=0.5)
    alg.update_step = alg.critic_model_start_update
    win, batch = _normalized_batch(alg)
    fb_kwargs, _, _, _ = alg._model_critic_inputs(win, batch)
    mask = fb_kwargs["negative_mask"].view(BATCH, 2, BATCH)
    assert fb_kwargs["negatives"].shape[0] == 2 * BATCH
    assert mask.diagonal(dim1=0, dim2=2).abs().sum() == 0
    alive = (batch.gammas.squeeze(-1) > 0).float()
    torch.testing.assert_close(mask[:, 0, :], (1 - torch.eye(BATCH)) * alive)
    losses, _ = alg._update_forward_backward(batch, **fb_kwargs)
    assert torch.isfinite(losses["Forward_Backward_Loss/fb_loss"])
