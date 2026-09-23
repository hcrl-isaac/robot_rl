# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import contextlib
import importlib
import math
import os
import pkgutil
import torch
import torch.nn as nn
import warnings
from collections.abc import Callable, Generator
from datetime import timedelta
from string import Template
from tensordict import TensorDict
from typing import TYPE_CHECKING, Any, TypeVar

import ot

import robot_rl

if TYPE_CHECKING:
    from robot_rl.modules import ParallelLayerNorm, ParallelLinear

T = TypeVar("T", TensorDict, torch.Tensor)


def get_param(param: Any, idx: int) -> Any:
    """Get a parameter for the given index.

    Args:
        param: Parameter or list/tuple of parameters.
        idx: Index to get the parameter for.
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param


@contextlib.contextmanager
def eval_mode(*modules: nn.Module) -> Generator[None, None, None]:
    """Context manager that temporarily switches modules to eval mode.

    On exit, each module is restored to its previous training state.
    """
    prev_states = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, was_training in zip(modules, prev_states, strict=True):
            m.train(was_training)


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from the name.

    Valid activation function names are: ``"elu"``, ``"selu"``, ``"relu"``, ``"crelu"``, ``"lrelu"``, ``"tanh"``,
    ``"sigmoid"``, ``"softplus"``, ``"gelu"``, ``"swish"``, ``"mish"``, ``"ball_norm"``, ``"identity"``.

    Args:
        act_name: Name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "ball_norm": _BallNorm(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(
    optimizer_name: str,
) -> type[torch.optim.Adam | torch.optim.AdamW | torch.optim.SGD | torch.optim.RMSprop]:
    """Resolve the optimizer from the name.

    Valid optimizer names are: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``.

    Args:
        optimizer_name: Name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a torch dtype from a string.

    Valid dtype names are: ``"float16"``, ``"bfloat16"``, ``"float32"``, ``"float64"``.
    """
    dtype_dict = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype_name = dtype_name.lower()
    if dtype_name in dtype_dict:
        return dtype_dict[dtype_name]
    else:
        raise ValueError(f"Invalid dtype '{dtype_name}'. Valid dtypes are: {list(dtype_dict.keys())}")


def resolve_callable(callable_or_name: type | Callable | str) -> Callable:
    """Resolve a callable from a string, type, or return callable directly.

    This function supports resolving callables from a direct callable input or from a string in one of these formats:

    - Direct callable: pass a type or function directly (for example, ``MyClass`` or ``my_func``).
    - Qualified name with colon: ``"module.path:Attr.Nested"`` (explicit, recommended).
    - Qualified name with dot: ``"module.path.ClassName"`` (implicit).
    - Simple name: for example ``"PPO"`` or ``"ActorCritic"`` (searched within ``robot_rl``).

    Args:
        callable_or_name: A callable (type/function) or string name.

    Returns:
        The resolved callable.

    Raises:
        TypeError: If input is neither a callable nor a string.
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute cannot be found in the module.
        ValueError: If a simple name cannot be found in robot_rl packages.
    """
    # Already a callable - return directly
    if callable(callable_or_name):
        return callable_or_name

    # Must be a string at this point
    if not isinstance(callable_or_name, str):
        raise TypeError(f"Expected callable or string, got {type(callable_or_name)}")

    # Handle qualified name with colon separator (e.g., "module.path:Attr.Nested")
    if ":" in callable_or_name:
        module_path, attr_path = callable_or_name.rsplit(":", 1)
        # Try to import the module
        module = importlib.import_module(module_path)
        # Try to get the attribute
        obj = module
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj  # type: ignore

    # Handle qualified name with dot separator (e.g., "module.path.ClassName")
    if "." in callable_or_name:
        parts = callable_or_name.split(".")
        module_found = False
        for i in range(len(parts) - 1, 0, -1):
            # Try to import the module with the first i parts
            module_path = ".".join(parts[:i])
            attr_parts = parts[i:]
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                continue
            module_found = True
            # Once a module is found, try to get the attribute
            obj = module
            try:
                for attr in attr_parts:
                    obj = getattr(obj, attr)
                return obj  # type: ignore
            except AttributeError:
                continue
        if module_found:
            raise AttributeError(f"Could not resolve '{callable_or_name}': attribute not found in module")
        raise ImportError(f"Could not resolve '{callable_or_name}': no valid module.attr split found")

    # Simple name - look for it in robot_rl
    for _, module_name, _ in pkgutil.iter_modules(robot_rl.__path__, "robot_rl."):
        module = importlib.import_module(module_name)
        if hasattr(module, callable_or_name):
            return getattr(module, callable_or_name)

    # Raise error if no approach worked
    raise ValueError(
        f"Could not resolve '{callable_or_name}'. Use qualified name like 'module.path:ClassName' "
        f"or pass the class directly."
    )


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """Validate the observation configuration and resolve missing observation sets.

    The input is an observation dictionary `obs` containing observation groups and a configuration dictionary
    `obs_groups` where the keys are the observation sets and the values are lists of observation groups.

    The configuration dictionary could for example look like::

        {
            "actor": ["group_1", "group_2"],
            "critic": ["group_1", "group_3"],
        }

    This means that the 'actor' observation set will contain the observations "group_1" and "group_2" and the 'critic'
    observation set will contain the observations "group_1" and "group_3". This function will check that all the
    observations in the 'actor' and 'critic' observation sets are present in the observation dictionary from the
    environment.

    Additionally, if one of the `default_sets`, e.g. "critic", is not present in the configuration dictionary, this
    function will:

    1. Check if a group with the same name exists in the observations and assign this group to the observation set.
    2. If 1. fails, it will assign the 'policy' observation group to the missing observation set.
    3. If 2. fails, an error is raised.

    Args:
        obs: Observations from the environment in the form of a dictionary.
        obs_groups: Dictionary mapping observation sets to lists of observation groups.
        default_sets: Default observation set names used by the algorithm. If not provided in ``obs_groups``, a
            default behavior gets triggered.

    Returns:
        The resolved observation groups.

    Raises:
        ValueError: If any observation set is an empty list.
        ValueError: If any observation set contains an observation term that is not present in the observations.
        ValueError: If a default observation set cannot be resolved according to the rules above.
    """
    # Check if obs_groups dictionary is empty
    if len(obs_groups) == 0:
        warnings.warn(
            "The observation configuration dictionary 'obs_groups' is empty and thus likely not configured. Consider"
            " configuring the 'obs_groups' dictionary explicitly"
        )
    else:
        # Check all observation sets for valid observation groups
        for set_name, groups in obs_groups.items():
            # Check if the list is empty
            if len(groups) == 0:
                raise ValueError(f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list.")
            # Check groups exist inside the observations from the environment
            for group in groups:
                if group not in obs:
                    raise ValueError(
                        f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                        f" environment. Available observations from the environment: {list(obs.keys())}"
                    )

    # Fill missing observation sets
    for default_set_name in default_sets:
        if default_set_name not in obs_groups:
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                    f" key. As an observation group with the name '{default_set_name}' was found, this is assumed to be"
                    f" the appropriate observation. Consider adding the '{default_set_name}' key to the 'obs_groups'"
                    f" dictionary for clarity. This behavior will be removed in a future version."
                )
            elif "policy" in obs:
                obs_groups[default_set_name] = ["policy"]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                    f" key. As an observation group with the name 'policy' was found, this is assumed to be the"
                    f" appropriate observation. Consider adding the '{default_set_name}' key to the 'obs_groups'"
                    f" dictionary for clarity. This behavior will be removed in a future version."
                )
            else:
                raise ValueError(
                    f"The observation configuration dictionary 'obs_groups' does not contain the '{default_set_name}'"
                    f" key and no suitable observation could be found in the observations from the environment."
                    f" Please refer to `robot_rl.utils.resolve_obs_groups()` for information on how to configure the"
                    f" 'obs_groups' dictionary correctly."
                )

    # Print the final parsed observation sets
    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups


def resolve_linear(
    in_features: int,
    out_features: int,
    num_parallel: int,
    bias: bool = True,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Linear | ParallelLinear:
    """Resolve and initialize linear module based on number of parallel networks.

    Returns:
        ParallelLinear module if using more than one parallel network; torch.nn.Linear otherwise.
    """
    if num_parallel > 1:
        from robot_rl.modules import ParallelLinear

        return ParallelLinear(in_features, out_features, num_parallel, bias=bias, device=device, dtype=dtype)
    else:
        return torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)


def resolve_layer_norm(
    normalized_shape: int,
    num_parallel: int,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    bias: bool = True,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.LayerNorm | ParallelLayerNorm:
    """Resolve and initialize layer norm module based on number of parallel networks.

    Returns:
        ParallelLayerNorm module if using more than one parallel network; torch.nn.LayerNorm otherwise.
    """
    if num_parallel > 1:
        from robot_rl.modules import ParallelLayerNorm

        return ParallelLayerNorm(
            normalized_shape,
            num_parallel,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    else:
        return torch.nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype
        )


def check_nan(obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Raise ``ValueError`` if any environment output contains NaN."""
    for key, tensor in obs.items():
        if torch.isnan(tensor).any():
            raise ValueError(
                f"The observation group '{key}' returned by the environment contains NaN values. This usually indicates"
                " a bug in the environment's step() or reset() function."
            )
    if torch.isnan(rewards).any():
        raise ValueError(
            "The rewards returned by the environment contain NaN values. This usually indicates a bug in the"
            " environment's reward computation."
        )
    if torch.isnan(dones).any():
        raise ValueError(
            "The dones returned by the environment contain NaN values. This usually indicates a bug in the"
            " environment's termination logic."
        )


def compile_model(model: torch.nn.Module, mode: str | None = None) -> torch.nn.Module:
    """Wrap a model with :func:`torch.compile`, validating the compile mode.

    Args:
        model: The model to compile.
        mode: The :func:`torch.compile` mode. CUDA-graph modes (``"reduce-overhead"``, ``"max-autotune"``) are rejected
        because they are incompatible with the multi-model forward patterns used by the algorithms (graph replay
        overwrites the previous call's output buffer). Use ``"default"`` or ``"max-autotune-no-cudagraphs"`` instead.
        Defaults to ``None``, in which case compilation is disabled.

    Returns:
        The compiled model, or the original model if ``mode`` is ``None``.

    Raises:
        ValueError: If ``mode`` is one of the unsupported CUDA-graph modes.
    """
    if mode is None:
        return model
    if mode in ("reduce-overhead", "max-autotune"):
        raise ValueError(
            f"torch_compile_mode='{mode}' uses CUDA graphs which are incompatible with the algorithms' multi-model "
            f"forward pattern. Use 'default' or 'max-autotune-no-cudagraphs', or set to None to disable."
        )
    return torch.compile(model, mode=mode)  # type: ignore


def split_and_pad_trajectories(tensor: T, dones: torch.Tensor) -> tuple[T, torch.Tensor]:
    """Split trajectories at done indices.

    Split trajectories, concatenate them and pad with zeros up to the length of the longest trajectory. Return masks
    corresponding to valid parts of the trajectories.

    Example (transposed for readability):
        Input: [[a1, a2, a3, a4 | a5, a6],
                [b1, b2 | b3, b4, b5 | b6]]

        Output:[[a1, a2, a3, a4], | [[True, True, True, True],
                [a5, a6, 0, 0],   |  [True, True, False, False],
                [b1, b2, 0, 0],   |  [True, True, False, False],
                [b3, b4, b5, 0],  |  [True, True, True, False],
                [b6, 0, 0, 0]]    |  [True, False, False, False]]

    Assumes that the input has the following order of dimensions: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have the order (num_envs, num_transitions_per_env, ...) for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # Split the tensor into trajectories
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # Add at least one full length trajectory
            trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], device=v.device))
            # Pad the trajectories to the length of the longest trajectory
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
            # Remove the added trajectory
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)], device=tensor.device
        )
    else:
        # Split the tensor into trajectories
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # Add at least one full length trajectory
        trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
        # Pad the trajectories to the length of the longest trajectory
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
        # Remove the added trajectory
        padded_trajectories = padded_trajectories[:, :-1]
    # Create masks for the valid parts of the trajectories
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories: T, masks: torch.Tensor) -> T:
    """Do the inverse operation of :meth:`split_and_pad_trajectories`."""
    # Need to transpose before and after the masking to have proper reshaping. Keep all trailing
    # feature dims (shape[2:]), not just the last, so >3D trajectories round-trip unchanged.
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], *trajectories.shape[2:])
        .transpose(1, 0)
    )  # type: ignore


@torch.jit.script
def compute_td_targets(data: torch.Tensor, lam: float) -> torch.Tensor:
    r"""Compute TD targets, i.e.

    .. math::

        mean(x) - \lam / (n^2 - n) * \sum_{i, j} |x_i - x_j|
    """
    n = data.shape[0]
    mean = data.mean(dim=0)
    uncertainty = torch.abs(data.unsqueeze(dim=0) - data.unsqueeze(dim=1)).sum(dim=(0, 1)) / (n**2 - n)
    return mean - lam * uncertainty


def _compute_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the distance matrix between two tensors."""
    x_norm = torch.sum(x**2, dim=-1, keepdim=True)
    y_norm = torch.sum(y**2, dim=-1, keepdim=True).transpose(-2, -1)
    mat = x_norm + y_norm - 2 * (x @ y.transpose(-2, -1))
    # ensure no negative values from numerical imprecision, then take sqrt for Euclidean distance
    return mat.clamp_(min=0.0).sqrt_()


def compute_emd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Earth Mover's Distance between two tensors."""
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError(
            f"Only 2D tensors are supported for EMD computation, but got x with {len(x.shape)} dimensions and y with "
            f"{len(y.shape)} dimensions."
        )
    cost_matrix = _compute_distance_matrix(x, y).detach()
    x_pot = torch.ones(x.shape[0], device=x.device) / x.shape[0]
    y_pot = torch.ones(y.shape[0], device=y.device) / y.shape[0]
    return ot.emd2(x_pot, y_pot, cost_matrix, numItermax=100000)  # type: ignore


def pad_to_size(x: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    """Zero-pad a tensor up to a provided size.

    Args:
        x: Tensor to pad
        size: Size of the padded tensor
        dim: Dimension along which to pad.
    """
    shape = list(x.shape)
    shape[dim] = max(size - shape[dim], 0)
    return torch.cat([x, torch.zeros(shape, device=x.device)], dim=dim)


def pad_to_size_repeat(x: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    """Pad a tensor up to a provided size by repeating its first slice.

    Padded entries are real data rather than zeros, for callers where zero is not a legal value.

    Args:
        x: Tensor to pad.
        size: Size of the padded tensor.
        dim: Dimension along which to pad.

    Returns:
        The padded tensor.
    """
    pad = max(size - x.shape[dim], 0)
    if pad == 0:
        return x
    first = x.narrow(dim, 0, 1)
    return torch.cat([x, first.expand(*[pad if i == dim else -1 for i in range(x.dim())])], dim=dim)


def forward_sliding_mean(x: torch.Tensor, window_len: int, dim: int = 0) -> torch.Tensor:
    """Smooth x by averaging within window_len.

    For each position ``t`` along ``dim``, returns the mean of ``x[..., t:t+window_len, ...]``,
    truncating at the end (so the last entries average over fewer elements).
    """
    # Move target dim to position 0 for cumsum
    perm = [i for i in range(x.dim()) if i != dim]
    perm.insert(0, dim)
    x = x.permute(perm)

    cumsum = torch.cumsum(x, dim=0)
    pad = torch.zeros(1, *cumsum.shape[1:], device=cumsum.device)
    cumsum = torch.cat([pad, cumsum], dim=0)

    length = x.shape[0]
    start_idx = torch.arange(length, device=x.device)
    end_idx = torch.clamp(start_idx + window_len, max=length)
    lengths = (end_idx - start_idx).view(length, *([1] * (x.dim() - 1)))

    mean = (cumsum[end_idx] - cumsum[start_idx]) / lengths
    inv_perm = [perm.index(i) for i in range(len(perm))]
    return mean.permute(inv_perm)


class _TimeDeltaTemplate(Template):
    delimiter = "%"


def format_date(fmt: str, time_s: float) -> str:
    """Convert a duration (in seconds) to the provided format.

    Args:
        fmt: Date format to use. Can use primitives %D (days), %H (hours), %M (minutes), and %S (seconds).
        time_s: Time to format, in seconds.
    """
    tdelta = timedelta(seconds=time_s)
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    # Zero-pad H, M, S
    d = {k: f"{v:02}" if k in ["H", "M", "S"] else str(v) for k, v in d.items()}
    t = _TimeDeltaTemplate(fmt)
    return t.substitute(**d)


def soft_update_params(
    params: tuple[torch.Tensor, ...],
    target_params: tuple[torch.Tensor, ...],
    tau: float,
) -> None:
    r"""Perform a soft update to target parameters.

    .. math::

        \theta' \leftarrow \tau \theta + (1 - \tau) \theta'

    where :math:`\theta` are the online parameters and :math:`\theta'` are the target parameters. A small ``tau``
    (e.g. 0.01) produces a slow-moving target network.

    Reference:
    - Lillicrap et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2019).
    """
    torch._foreach_mul_(target_params, 1.0 - tau)
    torch._foreach_add_(target_params, params, alpha=tau)


def demote_old_checkpoint(alg: Any, log_dir: str, it: int, keep: int | None, save_interval: int) -> int | None:
    """Rewrite the checkpoint that just left the keep-full window as policy-only, to bound storage.

    Demoted checkpoints keep only the keys from ``alg.policy_state_keys()`` -- still playable, eval'able,
    and exportable, but not training-resumable. No-op unless the algorithm exposes ``policy_state_keys``.

    Args:
        alg: The algorithm (must provide ``policy_state_keys`` for demotion to occur).
        log_dir: Directory holding the ``model_<it>.pt`` checkpoints.
        it: The iteration just saved.
        keep: How many recent checkpoints to keep full; ``None``/0 disables demotion.
        save_interval: Iterations between checkpoints (to locate the one leaving the window).

    Returns:
        The demoted iteration (so the caller can re-register the slimmed file with its logger), or None.
    """
    if not keep or not hasattr(alg, "policy_state_keys"):
        return None
    demote_it = it - keep * save_interval
    if demote_it < 0:
        return None
    path = os.path.join(log_dir, f"model_{demote_it}.pt")
    if not os.path.exists(path):
        return None
    # mmap so only the small policy tensors are read; the large resume-only tensors are never materialized
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if ckpt.get("_policy_only"):
        return None
    keep_keys = set(alg.policy_state_keys()) | {"iter", "env_step", "infos"}
    slim = {k: v for k, v in ckpt.items() if k in keep_keys}
    slim["_policy_only"] = True
    tmp_path = path + ".tmp"
    torch.save(slim, tmp_path)
    os.replace(tmp_path, path)
    return demote_it


class _BallNorm(nn.Module):
    """Ball norm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * nn.functional.normalize(x, dim=-1)
