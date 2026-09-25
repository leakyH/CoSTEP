import torch

REPLAY_POST_TRAIN_MODES = {"all_adaptor"}
from torch import nn
from torch.utils.data import DataLoader, Subset
from .mechanism import (
    DEFAULT_LATENCY_PROFILE,
    DEFAULT_LATENCY_SDLOG_MIN,
    get_latency_profile_config,
)

ADAPTOR_TAIL_LEN = 10
ADAPTOR_TAIL_VAL_LEN = 2
ADAPTOR_APPLY_MODES = {"adapt_train", "adapt_val", "test"}
ADAPTOR_TRAIN_MODES = {"train", "adapt_train"}


class SharedParamAffineAdaptor(nn.Module):
    """Shared scalar affine maps applied in probability logit space."""

    def __init__(self):
        super().__init__()
        self.beta_scale = nn.Parameter(torch.tensor(1.0))
        self.beta_bias = nn.Parameter(torch.tensor(0.0))
        self.gamma_scale = nn.Parameter(torch.tensor(1.0))
        self.gamma_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, param_b: torch.Tensor, param_g: torch.Tensor):
        eps = torch.finfo(param_b.dtype).eps
        beta_logit = torch.logit(torch.clamp(param_b, min=eps, max=1.0 - eps))
        gamma_logit = torch.logit(torch.clamp(param_g, min=eps, max=1.0 - eps))
        beta = torch.sigmoid(beta_logit * self.beta_scale + self.beta_bias)
        gamma = torch.sigmoid(gamma_logit * self.gamma_scale + self.gamma_bias)
        return (beta, gamma)


def _loader_indices(loader, mode: str):
    dataset = loader.dataset
    mode_indices = getattr(dataset, "mode_indices", None)
    if mode_indices is not None and mode in mode_indices:
        return [int(idx) for idx in mode_indices[mode]]
    return list(range(len(dataset)))


def _dataset_target_and_valid_mask(dataset):
    if isinstance(dataset, Subset):
        target, valid_mask = _dataset_target_and_valid_mask(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long, device=target.device)
        selected_target = target.index_select(0, indices)
        selected_mask = valid_mask.index_select(0, indices) if valid_mask is not None else None
        return (selected_target, selected_mask)
    output = getattr(dataset, "output", None)
    if output is not None:
        if isinstance(output, dict):
            return (output["target"], output.get("target_valid_mask"))
        return (output, None)
    tensors = getattr(dataset, "tensors", None)
    if tensors:
        return (tensors[-1], None)
    raise TypeError(f"cannot locate targets for dataset type {type(dataset).__name__}")


def _dataset_target(dataset):
    return _dataset_target_and_valid_mask(dataset)[0]


def _finite_target_mask(dataset, horizon: int | None = None):
    target, valid_mask = _dataset_target_and_valid_mask(dataset)
    target = torch.as_tensor(target)
    if valid_mask is not None:
        valid_mask = torch.as_tensor(valid_mask, device=target.device).bool()
    if horizon is not None and target.ndim >= 2:
        target = target[:, : int(horizon), ...]
        if valid_mask is not None:
            valid_mask = valid_mask[:, : int(horizon), ...]
    finite = torch.isfinite(target)
    if valid_mask is not None:
        finite &= valid_mask.expand_as(target)
    return finite


def count_finite_targets(loader, horizon: int | None = None) -> int:
    """Count scalar target entries available to a loader."""
    return int(_finite_target_mask(loader.dataset, horizon).sum().item())


def finite_label_lengths(loader, horizon: int | None = None) -> list[int]:
    """Return the number of finite forecast days for every loader origin."""
    finite = _finite_target_mask(loader.dataset, horizon)
    finite_by_day = finite.reshape(finite.shape[0], finite.shape[1], -1).any(dim=-1)
    return [int(value) for value in finite_by_day.sum(dim=1).tolist()]


def _subset_loader(loader, indices: list[int], batch_size: int, *, shuffle: bool):
    return DataLoader(
        dataset=Subset(loader.dataset, indices), batch_size=batch_size, shuffle=shuffle
    )


def _mode_split_audit(loader, absolute_indices, *, obs_len: int, pred_len: int):
    finite = _finite_target_mask(loader.dataset, pred_len)
    if int(finite.shape[0]) != len(absolute_indices):
        raise RuntimeError(
            f"post split audit origin/target mismatch: origins={len(absolute_indices)}, targets={finite.shape[0]}"
        )
    finite_by_day = finite.reshape(finite.shape[0], finite.shape[1], -1).any(dim=-1)
    rows = []
    target_days = set()
    for row_index, absolute_origin in enumerate(absolute_indices):
        horizons = torch.nonzero(finite_by_day[row_index], as_tuple=False).flatten().tolist()
        calendar_days = [int(absolute_origin) + int(obs_len) + int(horizon) for horizon in horizons]
        target_days.update(calendar_days)
        rows.append(
            {
                "absolute_origin": int(absolute_origin),
                "finite_label_length": int(len(horizons)),
                "finite_scalar_count": int(finite[row_index].sum().item()),
                "target_calendar_days": calendar_days,
            }
        )
    ordered_days = sorted(target_days)
    return {
        "origin_count": int(len(absolute_indices)),
        "absolute_origins": [int(index) for index in absolute_indices],
        "finite_scalar_count": int(finite.sum().item()),
        "finite_day_origin_count": int(finite_by_day.sum().item()),
        "target_calendar_day_count": int(len(ordered_days)),
        "target_calendar_days": ordered_days,
        "target_calendar_start": ordered_days[0] if ordered_days else None,
        "target_calendar_stop_inclusive": ordered_days[-1] if ordered_days else None,
        "origins": rows,
    }


def build_post_split_audit(loader_info: dict, *, obs_len: int, pred_len: int) -> dict:
    """Describe post split origins, finite labels, and target-calendar overlap."""
    mode_absolute_indices = {
        "train": loader_info.get("train_replay_absolute_indices", []),
        "validate": loader_info.get("validate_front_absolute_indices", []),
        "adapt_train": loader_info.get("adaptor_train_absolute_indices", []),
        "adapt_val": loader_info.get("adaptor_val_absolute_indices", []),
    }
    modes = {
        mode: _mode_split_audit(
            loader_info[mode], absolute_indices, obs_len=obs_len, pred_len=pred_len
        )
        for (mode, absolute_indices) in mode_absolute_indices.items()
        if loader_info.get(mode) is not None
    }
    pairwise = {}
    mode_names = list(modes)
    for left_index, left in enumerate(mode_names):
        left_days = set(modes[left]["target_calendar_days"])
        for right in mode_names[left_index + 1 :]:
            overlap = sorted(left_days.intersection(modes[right]["target_calendar_days"]))
            origin_overlap = sorted(
                set(modes[left]["absolute_origins"]).intersection(modes[right]["absolute_origins"])
            )
            pairwise[f"{left}__{right}"] = {
                "target_calendar_overlap_count": int(len(overlap)),
                "target_calendar_overlap_days": overlap,
                "origin_overlap_count": int(len(origin_overlap)),
                "origin_overlap_absolute_indices": origin_overlap,
            }
    embargo_len = int(loader_info.get("post_train_embargo_len", 0))
    tail_absolute_by_local = dict(
        zip(
            loader_info.get("adaptor_tail_indices", []),
            loader_info.get("adaptor_tail_absolute_indices", []),
        )
    )
    discarded_origin_groups = [
        {
            "reason": "validate_front_to_adapt_tail_embargo",
            "local_indices": loader_info.get("external_embargo_indices", []),
            "absolute_indices": loader_info.get("external_embargo_absolute_indices", []),
        },
        {
            "reason": "adapt_train_to_adapt_val_embargo",
            "local_indices": loader_info.get("internal_embargo_indices", []),
            "absolute_indices": loader_info.get("internal_embargo_absolute_indices", []),
        },
        {
            "reason": "below_minimum_finite_label_length",
            "local_indices": loader_info.get("ineligible_tail_indices", []),
            "absolute_indices": [
                tail_absolute_by_local[index]
                for index in loader_info.get("ineligible_tail_indices", [])
                if index in tail_absolute_by_local
            ],
        },
        {
            "reason": "original_train_replay_to_validate_front_embargo",
            "local_indices": [],
            "absolute_indices": loader_info.get("train_replay_dropped_absolute_indices", []),
        },
    ]
    return {
        "schema": "costep.post_split_audit.v1",
        "tail_len": int(loader_info.get("post_train_tail_len", 0)),
        "adapt_val_len": int(loader_info.get("post_train_adapt_val_len", 0)),
        "minimum_required_days": int(loader_info.get("post_train_min_val_label_len", 1)),
        "embargo_len": embargo_len,
        "partial_alltrain": bool(loader_info.get("post_train_partial_alltrain", False)),
        "tail_local_indices": loader_info.get("adaptor_tail_indices", []),
        "external_embargo_local_indices": loader_info.get("external_embargo_indices", []),
        "external_embargo_absolute_indices": loader_info.get(
            "external_embargo_absolute_indices", []
        ),
        "internal_embargo_local_indices": loader_info.get("internal_embargo_indices", []),
        "internal_embargo_absolute_indices": loader_info.get(
            "internal_embargo_absolute_indices", []
        ),
        "ineligible_tail_local_indices": loader_info.get("ineligible_tail_indices", []),
        "train_replay_dropped_absolute_indices": loader_info.get(
            "train_replay_dropped_absolute_indices", []
        ),
        "discarded_origin_groups": discarded_origin_groups,
        "modes": modes,
        "pairwise_target_calendar_overlap": pairwise,
        "blocked_zero_target_overlap": bool(
            embargo_len > 0
            and all((item["target_calendar_overlap_count"] == 0 for item in pairwise.values()))
        ),
        "base_checkpoint_history_overlap_not_reversed": bool(embargo_len > 0),
    }


def build_adaptor_tail_loader(
    base_data_loader: dict,
    batch_size: int,
    evaluation=False,
    prefer_existing_adapt_splits: bool = False,
    tail_len: int = ADAPTOR_TAIL_LEN,
    adapt_val_len: int = ADAPTOR_TAIL_VAL_LEN,
    pred_len: int | None = None,
    min_val_label_len: int = 1,
    include_short_in_adapt_train: bool = False,
    embargo_len: int = 0,
) -> dict:
    embargo_len = int(embargo_len)
    if embargo_len < 0:
        raise ValueError("post-training embargo length must be non-negative")
    if embargo_len > 0 and include_short_in_adapt_train:
        raise ValueError(
            "post-training embargo is incompatible with short-label adapt-train origins"
        )
    if (
        prefer_existing_adapt_splits
        and base_data_loader.get("adapt_train") is not None
        and (base_data_loader.get("adapt_val") is not None)
    ):
        if embargo_len > 0:
            raise ValueError(
                "post-training embargo cannot be inferred for pre-existing adapt splits"
            )
        adapt_train_indices = _loader_indices(base_data_loader["adapt_train"], "adapt_train")
        adapt_val_indices = _loader_indices(base_data_loader["adapt_val"], "adapt_val")
        validate_front_indices = (
            _loader_indices(base_data_loader["validate"], "validate")
            if base_data_loader.get("validate") is not None
            else []
        )
        result = {
            "train": base_data_loader["train"],
            "validate": base_data_loader.get("validate"),
            "adapt_train": base_data_loader["adapt_train"],
            "adapt_val": base_data_loader["adapt_val"],
            "test": base_data_loader.get("test"),
            "max_od": base_data_loader["max_od"],
            "node_scalers": base_data_loader.get("node_scalers"),
            "adaptor_tail_indices": adapt_train_indices + adapt_val_indices,
            "adaptor_train_indices": adapt_train_indices,
            "adaptor_val_indices": adapt_val_indices,
            "validate_front_indices": validate_front_indices,
            "train_replay_absolute_indices": _loader_indices(base_data_loader["train"], "train"),
            "validate_front_absolute_indices": validate_front_indices,
            "adaptor_train_absolute_indices": adapt_train_indices,
            "adaptor_val_absolute_indices": adapt_val_indices,
            "external_embargo_indices": [],
            "internal_embargo_indices": [],
            "ineligible_tail_indices": [],
            "post_train_tail_len": len(adapt_train_indices) + len(adapt_val_indices),
            "post_train_adapt_val_len": len(adapt_val_indices),
            "post_train_min_val_label_len": int(min_val_label_len),
            "post_train_embargo_len": 0,
            "post_train_partial_alltrain": bool(include_short_in_adapt_train),
            "post_val_label_lengths": finite_label_lengths(base_data_loader["adapt_val"], pred_len),
            "post_train_label_lengths": finite_label_lengths(
                base_data_loader["adapt_train"], pred_len
            ),
            "finite_target_counts": {
                mode: count_finite_targets(loader, pred_len)
                for (mode, loader) in (
                    ("train", base_data_loader["train"]),
                    ("validate", base_data_loader.get("validate")),
                    ("adapt_train", base_data_loader["adapt_train"]),
                    ("adapt_val", base_data_loader["adapt_val"]),
                    ("test", base_data_loader.get("test")),
                )
                if loader is not None
            },
        }
        for key in ("train_loss_state_mean", "train_loss_state_std"):
            if key in base_data_loader:
                result[key] = base_data_loader[key]
        return result
    validate_dataset = base_data_loader["validate"].dataset
    validate_len = len(validate_dataset)
    tail_len = min(int(tail_len), validate_len)
    if tail_len <= 0:
        raise ValueError("Cannot run adaptor post-training because the validate split is empty.")
    tail_start = validate_len - tail_len
    adapt_val_len = min(int(adapt_val_len), max(0, tail_len - 1))
    external_embargo_start = max(0, tail_start - embargo_len)
    validate_indices = list(range(0, external_embargo_start))
    external_embargo_indices = list(range(external_embargo_start, tail_start))
    target = _dataset_target(validate_dataset)
    horizon = int(pred_len) if pred_len is not None else int(target.shape[1])
    finite = torch.isfinite(torch.as_tensor(target)[:, :horizon, ...])
    finite_day_counts = finite.reshape(finite.shape[0], finite.shape[1], -1).any(dim=-1).sum(dim=1)
    eligible_tail_indices = [
        index
        for index in range(tail_start, validate_len)
        if int(finite_day_counts[index].item()) >= int(min_val_label_len)
    ]
    if len(eligible_tail_indices) <= adapt_val_len:
        raise ValueError(
            f"post tail has too few origins meeting the post-validation label length gate: eligible={len(eligible_tail_indices)}, adapt_val_len={adapt_val_len}, min_days={min_val_label_len}"
        )
    adapt_val_indices = eligible_tail_indices[-adapt_val_len:]
    adapt_val_set = set(adapt_val_indices)
    internal_embargo_start = max(tail_start, adapt_val_indices[0] - embargo_len)
    internal_embargo_indices = list(range(internal_embargo_start, adapt_val_indices[0]))
    internal_embargo_set = set(internal_embargo_indices)
    adapt_train_indices = [
        index
        for index in range(tail_start, validate_len)
        if index not in adapt_val_set
        and index not in internal_embargo_set
        and (embargo_len == 0 or index < internal_embargo_start)
        and (
            include_short_in_adapt_train
            or int(finite_day_counts[index].item()) >= int(min_val_label_len)
        )
    ]
    if not adapt_train_indices:
        raise ValueError("post tail has no eligible adapt-train origin")
    absolute_validate_indices = list(range(validate_len))
    mode_indices = getattr(validate_dataset, "mode_indices", None)
    if mode_indices is not None and "validate" in mode_indices:
        absolute_validate_indices = [int(index) for index in mode_indices["validate"]]
    validate_loader = _subset_loader(
        base_data_loader["validate"], validate_indices, batch_size, shuffle=False
    )
    adapt_train_loader = _subset_loader(
        base_data_loader["validate"],
        adapt_train_indices,
        batch_size,
        shuffle=False if evaluation else True,
    )
    adapt_val_loader = _subset_loader(
        base_data_loader["validate"], adapt_val_indices, batch_size, shuffle=False
    )
    train_loader = base_data_loader["train"]
    train_local_indices = list(range(len(train_loader.dataset)))
    train_absolute_indices = _loader_indices(train_loader, "train")
    train_replay_dropped_absolute_indices = []
    if embargo_len > 0:
        if len(train_local_indices) <= embargo_len:
            raise ValueError("post-training embargo leaves no original-train replay origins")
        train_replay_dropped_absolute_indices = train_absolute_indices[-embargo_len:]
        train_local_indices = train_local_indices[:-embargo_len]
        train_absolute_indices = train_absolute_indices[:-embargo_len]
        train_loader = _subset_loader(
            base_data_loader["train"],
            train_local_indices,
            batch_size,
            shuffle=False if evaluation else True,
        )
    result = {
        "train": train_loader,
        "validate": validate_loader,
        "adapt_train": adapt_train_loader,
        "adapt_val": adapt_val_loader,
        "test": base_data_loader.get("test"),
        "max_od": base_data_loader["max_od"],
        "node_scalers": base_data_loader.get("node_scalers"),
        "adaptor_tail_indices": list(range(tail_start, validate_len)),
        "adaptor_train_indices": adapt_train_indices,
        "adaptor_val_indices": adapt_val_indices,
        "validate_front_indices": validate_indices,
        "external_embargo_indices": external_embargo_indices,
        "internal_embargo_indices": internal_embargo_indices,
        "ineligible_tail_indices": [
            index for index in range(tail_start, validate_len) if index not in eligible_tail_indices
        ],
        "train_replay_absolute_indices": train_absolute_indices,
        "train_replay_dropped_absolute_indices": train_replay_dropped_absolute_indices,
        "adaptor_tail_absolute_indices": [
            absolute_validate_indices[index] for index in range(tail_start, validate_len)
        ],
        "adaptor_train_absolute_indices": [
            absolute_validate_indices[index] for index in adapt_train_indices
        ],
        "adaptor_val_absolute_indices": [
            absolute_validate_indices[index] for index in adapt_val_indices
        ],
        "validate_front_absolute_indices": [
            absolute_validate_indices[index] for index in validate_indices
        ],
        "external_embargo_absolute_indices": [
            absolute_validate_indices[index] for index in external_embargo_indices
        ],
        "internal_embargo_absolute_indices": [
            absolute_validate_indices[index] for index in internal_embargo_indices
        ],
        "post_val_label_lengths": [
            int(finite_day_counts[index].item()) for index in adapt_val_indices
        ],
        "post_train_label_lengths": [
            int(finite_day_counts[index].item()) for index in adapt_train_indices
        ],
        "post_train_min_val_label_len": int(min_val_label_len),
        "post_train_tail_len": int(tail_len),
        "post_train_adapt_val_len": int(adapt_val_len),
        "post_train_embargo_len": int(embargo_len),
        "post_train_partial_alltrain": bool(include_short_in_adapt_train),
        "finite_target_counts": {
            "train": count_finite_targets(train_loader, pred_len),
            "validate": count_finite_targets(validate_loader, pred_len),
            "adapt_train": count_finite_targets(adapt_train_loader, pred_len),
            "adapt_val": count_finite_targets(adapt_val_loader, pred_len),
            **(
                {"test": count_finite_targets(base_data_loader["test"], pred_len)}
                if base_data_loader.get("test") is not None
                else {}
            ),
        },
    }
    for key in ("train_loss_state_mean", "train_loss_state_std"):
        if key in base_data_loader:
            result[key] = base_data_loader[key]
    return result


def model_has_adaptor(model):
    return hasattr(model, "param_adaptor")


def ensure_linear_param_adaptor(model) -> SharedParamAffineAdaptor:
    device, dtype = _model_device_dtype(model)
    current = getattr(model, "param_adaptor", None)
    if not isinstance(current, SharedParamAffineAdaptor):
        model.param_adaptor = SharedParamAffineAdaptor().to(device=device, dtype=dtype)
    else:
        model.param_adaptor.to(device=device, dtype=dtype)
    return model.param_adaptor


def ensure_adaptor_lognormal_latency_params(
    model, meanlog: torch.Tensor = None, raw_sdlog: torch.Tensor = None, requires_grad: bool = True
):
    if not hasattr(model, "latency_window_length"):
        return False
    device, dtype = _model_device_dtype(model)
    if meanlog is None:
        if hasattr(model, "latency_meanlog"):
            meanlog = model.latency_meanlog.detach().clone()
        else:
            config = get_latency_profile_config(DEFAULT_LATENCY_PROFILE)
            meanlog = torch.tensor(float(config["meanlog"]))
    if raw_sdlog is None:
        if hasattr(model, "latency_raw_sdlog"):
            raw_sdlog = model.latency_raw_sdlog.detach().clone()
        else:
            config = get_latency_profile_config(DEFAULT_LATENCY_PROFILE)
            sdlog = torch.tensor(float(config["sdlog"]), device=device, dtype=dtype)
            raw_sdlog = model._inverse_softplus(
                sdlog - getattr(model, "latency_sdlog_min", DEFAULT_LATENCY_SDLOG_MIN)
            )
    meanlog = meanlog.to(device=device, dtype=dtype)
    raw_sdlog = raw_sdlog.to(device=device, dtype=dtype)
    if hasattr(model, "adaptor_latency_meanlog") and isinstance(
        model.adaptor_latency_meanlog, nn.Parameter
    ):
        model.adaptor_latency_meanlog.data.copy_(meanlog)
    else:
        model.adaptor_latency_meanlog = nn.Parameter(meanlog.clone().detach())
    if hasattr(model, "adaptor_latency_raw_sdlog") and isinstance(
        model.adaptor_latency_raw_sdlog, nn.Parameter
    ):
        model.adaptor_latency_raw_sdlog.data.copy_(raw_sdlog)
    else:
        model.adaptor_latency_raw_sdlog = nn.Parameter(raw_sdlog.clone().detach())
    model.adaptor_latency_meanlog.requires_grad = requires_grad
    model.adaptor_latency_raw_sdlog.requires_grad = requires_grad
    return True


def remove_adaptor_lognormal_latency_params(model):
    for name in ["adaptor_latency_meanlog", "adaptor_latency_raw_sdlog"]:
        if hasattr(model, name):
            delattr(model, name)


def remove_adaptor_latency_state(model):
    remove_adaptor_lognormal_latency_params(model)


def _matches_latency_param(name: str) -> bool:
    latency_tokens = (
        "latency_meanlog",
        "latency_raw_sdlog",
    )
    return any((token in name for token in latency_tokens))


def _matches_base_latency_param(name: str) -> bool:
    return _matches_latency_param(name) and (not name.startswith("adaptor_latency_"))


def _model_device_dtype(model):
    for tensor in list(model.parameters()) + list(model.buffers()):
        return (tensor.device, tensor.dtype)
    return (torch.device("cpu"), torch.float32)


def _set_base_latency_requires_grad(model, requires_grad: bool) -> int:
    count = 0
    for name, param in model.named_parameters():
        if _matches_base_latency_param(name):
            param.requires_grad = requires_grad
            count += 1
    return count


def enable_adaptor_latency_finetune(model):
    converted = ensure_adaptor_lognormal_latency_params(model, requires_grad=True)
    _set_base_latency_requires_grad(model, False)
    return converted


def mode_uses_adaptor(mode: str, stage: str = None) -> bool:
    if stage in [None, "base"]:
        return False
    return mode in ADAPTOR_APPLY_MODES


def mode_is_train(mode: str) -> bool:
    return mode in ADAPTOR_TRAIN_MODES


def configure_adaptor_training(model, mode="all_adaptor", profile="finetune"):
    dynamic = getattr(model, 'sir_decoder_variant', 'delay') == 'dynamic'
    expected_profile = 'freeze' if dynamic else 'finetune'
    if mode != "all_adaptor" or profile != expected_profile:
        raise ValueError(f"This decoder requires all_adaptor with {expected_profile}")
    model._base_latency_method = model.latency_method
    model._post_latency_policy = profile
    model._post_latency_profile = None
    model._post_latency_empirical_to_lognormal = False
    ensure_linear_param_adaptor(model)
    remove_adaptor_latency_state(model)
    for param in model.parameters():
        param.requires_grad = True
    if not dynamic:
        model._post_latency_empirical_to_lognormal = enable_adaptor_latency_finetune(model)
    trainable = [(name, param) for (name, param) in model.named_parameters() if param.requires_grad]
    return {"mode": mode, "profile": profile, "trainable_params": trainable}
