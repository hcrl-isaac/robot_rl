"""Rebuild trained policies from a checkpoint + train cfg (no env needed) for export.

Every rebuilt model has its observation normalizer baked into the model's own ``obs_normalizer``
slot, so it exports through the standard ``as_jit()``/``as_onnx()`` API regardless of algorithm.
"""

from __future__ import annotations

import copy
import os
import re
import torch
import torch.nn as nn

from robot_rl.models.cnn_rnn_model import CNNRNNModel
from robot_rl.models.inference import EncoderInferencePolicy
from robot_rl.models.rnn_model import RNNModel
from robot_rl.modules.distribution import Distribution, bound_by_clip_actions
from robot_rl.utils.utils import resolve_callable

DEFAULT_ONNX_OPSET = 18


def _load_bn(nsd: dict[str, torch.Tensor], keys: list[str]) -> nn.BatchNorm1d:
    """Rebuild ONE BatchNorm1d covering the concatenation of the given obs groups.

    BatchNorm statistics are per-feature, so concatenating the per-group running stats is exactly
    equivalent to applying the per-group normalizers and concatenating.
    """
    prefix = "modules_dict." if any(k.startswith("modules_dict.") for k in nsd) else ""
    means = torch.cat([nsd[f"{prefix}{k}.running_mean"] for k in keys])
    variances = torch.cat([nsd[f"{prefix}{k}.running_var"] for k in keys])
    bn = nn.BatchNorm1d(means.shape[0], momentum=0.01, affine=False)
    bn.load_state_dict({
        "running_mean": means,
        "running_var": variances,
        "num_batches_tracked": nsd[f"{prefix}{keys[0]}.num_batches_tracked"],
    })
    return bn


def _group_dims(nsd: dict[str, torch.Tensor]) -> dict[str, int]:
    """Obs-group name -> flat dimension, read from the normalizer running stats."""
    dims = {}
    for k, v in nsd.items():
        if k.endswith(".running_mean"):
            dims[k.removeprefix("modules_dict.").removesuffix(".running_mean")] = v.shape[0]
    return dims


def _distribution_class(dist_cfg: dict) -> type[Distribution]:
    """Return the distribution class a logged ``distribution_cfg`` names."""
    if "class_name" not in dist_cfg:
        raise ValueError("The logged distribution_cfg has no 'class_name'; it must name its distribution class.")
    return resolve_callable(dist_cfg["class_name"])


def _num_actions(model_cfg: dict, sd: dict[str, torch.Tensor], default_class: str = "MLPModel") -> int:
    """Return a model's action dimension from its head width and its distribution's input layout.

    Args:
        model_cfg: The logged model cfg, naming its class and its distribution.
        sd: The model's state dict.
        default_class: Model class when the cfg names none.
    """
    width = resolve_callable(model_cfg.get("class_name", default_class)).head_output_width(sd)
    dist_cfg = model_cfg.get("distribution_cfg")
    return width if dist_cfg is None else _distribution_class(dist_cfg).output_dim_for_input_width(width)


def _extend_bn_identity(bn: nn.BatchNorm1d, extra_dims: int) -> nn.BatchNorm1d:
    """Extend a BatchNorm1d with identity statistics (mean 0, var 1) over trailing features.

    For models whose export forward takes side inputs concatenated after the obs.
    """
    if extra_dims == 0:
        return bn
    out = nn.BatchNorm1d(bn.num_features + extra_dims, momentum=0.01, affine=False)
    out.running_mean[: bn.num_features] = bn.running_mean
    out.running_var[: bn.num_features] = bn.running_var
    out.running_mean[bn.num_features :] = 0.0
    out.running_var[bn.num_features :] = 1.0
    out.num_batches_tracked.copy_(bn.num_batches_tracked)
    return out


def bake_live_normalizer(model: nn.Module, normalizers: nn.Module) -> nn.Module:
    """Bake an algorithm's live per-group normalizer into a copy of ``model``.

    The copy exports via the standard ``as_jit()``/``as_onnx()`` API.
    """
    groups = list(model.obs_groups)
    means = torch.cat([normalizers.modules_dict[g].running_mean for g in groups])
    variances = torch.cat([normalizers.modules_dict[g].running_var for g in groups])
    bn = nn.BatchNorm1d(means.shape[0], momentum=0.01, affine=False)
    bn.running_mean.copy_(means)
    bn.running_var.copy_(variances)
    return bake_normalizer(model, bn)


def bake_normalizer(model: nn.Module, normalizer: nn.Module) -> nn.Module:
    """Return a copy of ``model`` with ``normalizer`` installed as its own obs normalizer.

    The standard ``as_jit()``/``as_onnx()`` exports then include it.
    """
    model = copy.deepcopy(model).eval()
    model.obs_normalizer = copy.deepcopy(normalizer).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _first_mlp_input_dim(sd: dict[str, torch.Tensor]) -> int:
    """Width of a model's first MLP layer, i.e. everything its trunk consumes."""
    first_idx = min(int(m.group(1)) for k in sd if (m := re.match(r"mlp\.(\d+)\.weight", k)))
    return sd[f"mlp.{first_idx}.weight"].shape[1]


def _rebuild_ppo(train_cfg: dict, ckpt: dict, all_models: bool) -> dict[str, nn.Module]:
    if "memory_state_dict" in ckpt:
        raise NotImplementedError("Export of recurrent (memory-bearing) policies is not supported.")
    cfg = copy.deepcopy(train_cfg)
    models: dict[str, nn.Module] = {}

    # an encoder head's first layer is ``obs_dim + latent_dim`` wide; subtract the latent to size its obs
    encoder_cfg = cfg.get("algorithm", {}).get("encoder_cfg")
    latent_dim = int(encoder_cfg["output_dim"]) if encoder_cfg is not None else 0

    def build_from_groups(
        model_cfg: dict, sd: dict[str, torch.Tensor], obs_set: str, output_dim: int, other_dims: tuple[int, ...]
    ) -> nn.Module:
        model_cfg = dict(model_cfg)
        model_class = resolve_callable(model_cfg.pop("class_name", "MLPModel"))
        obs_dim = _first_mlp_input_dim(sd) - sum(other_dims)
        groups = cfg["obs_groups"][obs_set]
        # only the concatenated dim matters for layer sizes; put it all on the first group
        obs = {g: torch.zeros(1, obs_dim if i == 0 else 0) for i, g in enumerate(groups)}
        model = model_class(obs, {obs_set: groups}, obs_set, output_dim, other_input_dims=other_dims, **model_cfg)
        model.load_state_dict(sd, strict=True)
        return model.eval()

    def build(name: str, sd_key: str, output_dim: int) -> nn.Module:
        obs_set = "actor" if name == "policy" else "critic"
        other_dims = (latent_dim,) if latent_dim else ()
        return build_from_groups(
            cfg[name if name != "policy" else "actor"], ckpt[sd_key], obs_set, output_dim, other_dims
        )

    encoder: nn.Module | None = None
    if encoder_cfg is not None:
        if "encoder_state_dict" not in ckpt:
            raise ValueError("The train cfg configures an encoder but the checkpoint has no 'encoder_state_dict'.")
        encoder = build_from_groups(encoder_cfg["model"], ckpt["encoder_state_dict"], "encoder", latent_dim, ())

    models["policy"] = build("policy", "actor_state_dict", _num_actions(cfg["actor"], ckpt["actor_state_dict"]))
    if all_models:
        if "critic_state_dict" not in ckpt:
            # SAC checkpoints a pair of fused critics, whose constructor this builder does not fit
            raise NotImplementedError("all_models is not supported for this checkpoint; its policy exports alone.")
        models["critic"] = build("critic", "critic_state_dict", 1)
    if encoder is not None:
        # fold the encoder in, so the export takes one ``[head_obs ; encoder_obs]`` input
        models = {name: EncoderInferencePolicy(encoder, model) for name, model in models.items()}
    return models


def _rebuild_distillation(
    train_cfg: dict, ckpt: dict, obs_shapes: dict[str, tuple[int, ...]] | None = None
) -> dict[str, nn.Module]:
    """Rebuild a distillation student from its checkpoint.

    Args:
        train_cfg: The run's train cfg, holding the ``student`` model cfg and its obs groups.
        ckpt: The checkpoint, holding ``student_state_dict`` with its baked observation normalizer.
        obs_shapes: ``(channels, height, width)`` of each image group, which a checkpoint does not record;
            optionally ``(dim,)`` of each 1D group, needed when the student has no observation normalizer.

    Returns:
        The student under the ``policy`` export name.
    """
    cfg = copy.deepcopy(train_cfg)
    sd = ckpt["student_state_dict"]
    model_cfg = dict(cfg["student"])
    model_class = resolve_callable(model_cfg.pop("class_name", "MLPModel"))
    if issubclass(model_class, RNNModel) and not issubclass(model_class, CNNRNNModel):
        raise NotImplementedError("Export of recurrent (memory-bearing) students is not supported.")
    groups = cfg["obs_groups"]["student"]
    shapes = {g: tuple(obs_shapes[g]) for g in groups if obs_shapes and g in obs_shapes}
    images = {g: shape for g, shape in shapes.items() if len(shape) == 3}
    if "cnn_cfg" in model_cfg and not images:
        raise ValueError(f"{model_class.__name__} encodes images: give each image group's (C, H, W) in obs_shapes")
    flat = [g for g in groups if g not in images]
    if all(g in shapes for g in flat):
        dims = {g: shapes[g][0] for g in flat}
    else:
        if not images:
            total = _first_mlp_input_dim(sd)
        elif "obs_normalizer._mean" in sd:
            total = sd["obs_normalizer._mean"].shape[-1]
        else:
            raise ValueError("this image student records no 1D input size: give each 1D group's (dim,) in obs_shapes")
        # only the concatenated 1D dim matters for layer sizes; put it all on the first 1D group
        dims = {g: total if g == flat[0] else 0 for g in flat}
    obs = {g: torch.zeros(1, dims[g]) for g in flat}
    obs |= {g: torch.zeros(1, *shape) for g, shape in images.items()}
    num_actions = _num_actions(cfg["student"], sd)
    model = model_class(obs, {"student": groups}, "student", num_actions, **model_cfg)
    model.load_state_dict(sd, strict=True)
    return {"policy": model.eval()}


# FB-CPR models: name -> (cfg key, obs set, default class, output spec, other-input spec)
_FBCPR_MODELS = {
    "policy": ("actor", "actor", "ResidualFuseModel", "num_actions", ("z_dim",)),
    "backward": ("backward_map", "backward", "MLPModel", "z_dim", ()),
    "forward": ("forward_map", "critic", "ResidualFuseModel", "z_dim", ("z_dim", "num_actions")),
    "disc_critic": ("disc_critic", "critic", "ResidualFuseModel", 1, ("z_dim", "num_actions")),
    "aux_critic": ("aux_critic", "critic", "ResidualFuseModel", 1, ("z_dim", "num_actions")),
    "discriminator": ("discriminator", "discriminator", "MLPModel", 1, ("z_dim",)),
}


def _rebuild_fbcpr(train_cfg: dict, ckpt: dict, all_models: bool) -> dict[str, nn.Module]:
    cfg = copy.deepcopy(train_cfg)
    nsd = ckpt["obs_normalizer_state_dict"]
    obs = {group: torch.zeros(1, dim) for group, dim in _group_dims(nsd).items()}
    num_actions = _num_actions(cfg["actor"], ckpt["actor_state_dict"], default_class=_FBCPR_MODELS["policy"][2])
    dims = {"z_dim": cfg["algorithm"]["z_dim"], "num_actions": num_actions}

    def build(name: str) -> nn.Module:
        cfg_key, obs_set, default_class, out_spec, other_spec = _FBCPR_MODELS[name]
        # the actor sits at the top of the runner cfg, the FB maps and critics under ``algorithm``
        model_cfg = dict(cfg[cfg_key] if cfg_key in cfg else cfg["algorithm"][cfg_key])
        model_class = resolve_callable(model_cfg.pop("class_name", default_class))
        dist_cfg = model_cfg.get("distribution_cfg")
        # the same bounds rule as training; any other head keeps the bounds its cfg dumped
        if dist_cfg is not None and not bound_by_clip_actions(dist_cfg, cfg["clip_actions"]):
            dist_cfg.setdefault("low", -cfg["clip_actions"])
            dist_cfg.setdefault("high", cfg["clip_actions"])
        out_dim = dims[out_spec] if isinstance(out_spec, str) else out_spec
        other_dims = tuple(dims[k] for k in other_spec)
        bn = _load_bn(nsd, cfg["obs_groups"][obs_set])
        if name == "policy":
            model = model_class(obs, cfg["obs_groups"], obs_set, (other_dims[0], 0), out_dim, **model_cfg)
        elif name in ("backward", "discriminator"):
            model = model_class(obs, cfg["obs_groups"], obs_set, out_dim, other_input_dims=other_dims, **model_cfg)
            # MLPModel exports take ONE concatenated input (obs + side inputs): extend the baked
            # normalizer with identity stats over the side-input dims
            bn = _extend_bn_identity(bn, sum(other_dims))
            model.obs_dim = bn.num_features  # sizes the export dummy inputs
        else:
            model = model_class(obs, cfg["obs_groups"], obs_set, other_dims, out_dim, **model_cfg)
        model.load_state_dict(ckpt[f"{cfg_key}_state_dict"], strict=True)
        return bake_normalizer(model, bn)

    names = list(_FBCPR_MODELS) if all_models else ["policy"]
    return {name: build(name) for name in names}


def rebuild_models(
    train_cfg: dict, ckpt: dict, all_models: bool = False, obs_shapes: dict[str, tuple[int, ...]] | None = None
) -> dict[str, nn.Module]:
    """Rebuild the trained models from a checkpoint, normalizers baked in; keyed by export name.

    Args:
        train_cfg: The run's train cfg.
        ckpt: The loaded checkpoint.
        all_models: Also rebuild the models that are not the policy, where the algorithm has them.
        obs_shapes: ``(channels, height, width)`` of each image observation group, and optionally ``(dim,)`` of
            each 1D group, for image students.
    """
    if "backward_map_state_dict" in ckpt:
        return _rebuild_fbcpr(train_cfg, ckpt, all_models)
    if "student_state_dict" in ckpt:
        return _rebuild_distillation(train_cfg, ckpt, obs_shapes)
    return _rebuild_ppo(train_cfg, ckpt, all_models)


def save_jit(module: nn.Module, path: str, filename: str) -> str:
    """Save a TorchScript export (modules with ``jit_trace`` set are traced; FuseModels don't script)."""
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with torch.no_grad():
        if getattr(module, "jit_trace", False):
            scripted = torch.jit.trace(module, module.get_dummy_inputs())
        else:
            scripted = torch.jit.script(module)
    scripted.save(save_path)
    return save_path


def save_onnx(
    module: nn.Module, path: str, filename: str, verbose: bool = False, opset: int = DEFAULT_ONNX_OPSET
) -> str:
    """Save an ONNX export; the module must provide dummy inputs and input/output names.

    Args:
        module: Export-ready module (``as_onnx()`` output) with ``get_dummy_inputs``/``input_names``/``output_names``.
        path: Directory to write into (created if missing).
        filename: File name within ``path``.
        verbose: Forward to ``torch.onnx.export``.
        opset: ONNX opset to target; raise it for modules that need a newer operator.

    Returns:
        The saved file path.
    """
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    torch.onnx.export(
        module,
        module.get_dummy_inputs(),
        save_path,
        export_params=True,
        opset_version=opset,
        verbose=verbose,
        input_names=module.input_names,
        output_names=module.output_names,
    )
    return save_path
