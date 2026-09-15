from __future__ import annotations
import json
import copy
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from .reporting import (
    FrozenGapProfile,
    as_tnc,
    build_reporting_metadata,
    coarsen_reporting_calendar,
)
from .profiles import PROFILES, PROTOCOL

DEFAULT_PROFILE_LEVEL = "L2"
PROFILE_LEVELS = ("L0", "L1", "L2", "L3", "L4")
SPLIT_STREAM_IDS = {"train": 0, "validate": 1, "test": 2}


def load_japan_irregular_protocol():
    return copy.deepcopy(PROTOCOL)


def load_japan_irregular_profile(level="L2"):
    return FrozenGapProfile.from_payload(PROFILES[normalize_japan_irregular_level(level)])


def normalize_japan_irregular_level(level: str) -> str:
    normalized = str(level).upper()
    if normalized not in PROFILE_LEVELS:
        raise ValueError(
            f"unknown JAPAN-Irregular level {level!r}; expected one of {PROFILE_LEVELS}"
        )
    return normalized


def identify_japan_irregular_profile_level(profile):
    for level in PROFILE_LEVELS:
        if profile == load_japan_irregular_profile(level):
            return level
    raise ValueError("Unknown JAPAN-Irregular gap distribution")


def derive_split_seed(data_seed: int, split: str) -> int:
    if split not in SPLIT_STREAM_IDS:
        raise ValueError(f"unknown JAPAN-Irregular split {split!r}")
    sequence = np.random.SeedSequence([int(data_seed), SPLIT_STREAM_IDS[split]])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def synchronize_japan_sir_reports(reference_sir, daily_cases, observed_cases, report_mask):
    """Release true case and removed increments together, preserving initial stocks.

    R is cumulative removed/recovered data. Its signed revisions must survive;
    a negative revision is not a negative case report. The unavailable pre-series
    R history remains in the initial stock (day-zero exit increment is zero).
    """
    sir = np.asarray(reference_sir)
    cases = as_tnc(daily_cases).astype(np.float64, copy=False)
    observed = as_tnc(observed_cases).astype(np.float64, copy=False)
    mask = as_tnc(report_mask).astype(bool, copy=False)
    if sir.shape[:2] != cases.shape[:2] or sir.shape[-1] != 3 or observed.shape != cases.shape or mask.shape != cases.shape:
        raise ValueError('synchronized JAPAN cases, reports, mask and SIR must align')
    if not np.isfinite(sir).all():
        raise ValueError('non-finite original JAPAN SIR')
    exits = np.diff(sir[..., [2]], axis=0, prepend=sir[:1, :, [2]]).astype(np.float64)
    # SIR reconstruction requires the source states.
    if not np.allclose(np.diff(sir[..., 0], axis=0), -cases[1:, :, 0], rtol=0, atol=1e-6):
        raise ValueError('original S increments do not match daily cases')
    if not np.allclose(np.diff(sir[..., 1], axis=0), cases[1:, :, 0] - exits[1:, :, 0], rtol=0, atol=1e-6):
        raise ValueError('original I increments do not match daily cases minus R increments')
    released_exits = np.zeros_like(exits)
    pending = np.zeros_like(exits[0])
    for t in range(len(exits)):
        pending += exits[t]
        released_exits[t] = np.where(mask[t], pending, 0.0)
        pending = np.where(mask[t], 0.0, pending)
    if not mask[-1].all():
        raise ValueError('report calendar must flush every node at the series end')
    initial_s = sir[0, :, 0] + cases[0, :, 0]
    initial_i = sir[0, :, 1] - cases[0, :, 0]
    cumulative_cases = np.cumsum(observed[..., 0], axis=0)
    cumulative_exits = np.cumsum(released_exits[..., 0], axis=0)
    rebuilt = np.stack((initial_s - cumulative_cases,
                        initial_i + cumulative_cases - cumulative_exits,
                        sir[0, :, 2] + cumulative_exits), axis=-1).astype(sir.dtype)
    if not np.allclose(released_exits.sum(0), exits.sum(0), rtol=0, atol=1e-6):
        raise AssertionError('removed counts were not conserved')
    np.testing.assert_allclose(rebuilt.sum(-1), sir.sum(-1), rtol=0, atol=1e-6)
    return rebuilt, exits, released_exits



def build_interval_node_features(
    data_node: np.ndarray, reporting_metadata: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the canonical interval-aware node view used by protocol-v1 consumers."""
    interval_node = np.array(data_node, copy=True)
    report_mask = np.asarray(reporting_metadata["report_mask"], dtype=bool)
    report_rate = np.asarray(reporting_metadata["report_rate"])
    report_interval = np.asarray(reporting_metadata["report_interval"])
    interval_node[..., 0] = np.where(report_mask, np.log1p(report_rate), 0.0)[..., 0]
    interval_node = np.concatenate(
        [
            interval_node,
            report_mask.astype(interval_node.dtype),
            np.log1p(report_interval).astype(interval_node.dtype),
        ],
        axis=-1,
    )
    names = tuple(
        [
            "log1p_report_rate",
            *[
                f"source_node_feature_{index}"
                for index in range(1, np.asarray(data_node).shape[-1])
            ],
            "report_mask",
            "log1p_report_interval",
        ]
    )
    return (interval_node, names)


def build_japan_irregular_view(
    data: Mapping,
    *,
    split_ranges: Mapping[str, Sequence[int]],
    profile: FrozenGapProfile,
    data_seed: int,
    coarsen_train: bool,
    profile_level: str | None = None,
    observation_length: int = 14,
    forecast_horizon: int = 14,
) -> tuple[dict, dict]:
    """Build reporting views with synchronized exits and continuous states."""
    if set(split_ranges) != {"train", "validate", "test"}:
        raise ValueError("split_ranges must contain train/validate/test")
    reference = as_tnc(data["y"], name="data['y']")
    observed = np.array(reference, copy=True)
    report_mask = np.ones(reference.shape, dtype=bool)
    manifests = {}
    coarsened = {"validate", "test"} | ({"train"} if coarsen_train else set())
    for split in ("train", "validate", "test"):
        start, stop = map(int, split_ranges[split])
        if split not in coarsened:
            manifests[split] = {"mode": "native", "start": start, "stop": stop, "seed": None}
            continue
        split_seed = derive_split_seed(data_seed, split)
        split_y, split_mask, manifest = coarsen_reporting_calendar(
            reference, start=start, stop=stop, profile=profile, seed=split_seed
        )
        observed[start:stop] = split_y[start:stop]
        report_mask[start:stop] = split_mask[start:stop]
        manifests[split] = {
            "mode": "coarsened",
            "data_seed": int(data_seed),
            "derived_seed": split_seed,
            **manifest,
        }
    rebuilt, daily_exits, reported_exits = synchronize_japan_sir_reports(
        data["SIR"], reference, observed, report_mask
    )
    reset_indices = tuple((int(split_ranges[name][0]) for name in ("train", "validate", "test")))
    view = dict(data)
    view.update(
        {
            "reference_daily_y": np.array(reference, copy=True),
            "reference_daily_exits": daily_exits,
            "observed_report_exits": reported_exits,
            "y": observed,
            "report_mask": report_mask,
            "SIR": rebuilt,
            "reporting_reset_indices": reset_indices,
        }
    )
    node = np.array(data["node"], copy=True)
    node[..., 0] = np.log1p(observed[..., 0])
    view["node"] = node
    reporting_metadata = build_reporting_metadata(
        observed, report_mask=report_mask, reset_indices=reset_indices
    )
    interval_node, interval_names = build_interval_node_features(node, reporting_metadata)
    view["reporting_metadata"] = reporting_metadata
    view["interval_node"] = interval_node
    view["interval_node_feature_names"] = interval_names
    selected_level = (
        identify_japan_irregular_profile_level(profile)
        if profile_level is None
        else normalize_japan_irregular_level(profile_level)
    )
    if profile != load_japan_irregular_profile(selected_level):
        raise ValueError("Gap distribution does not match the selected level")
    manifest = {
        "schema": "costep.japan_irregular.v2",
        "state_construction": "synchronized_observed_exits_continuous_stocks",
        "profile_level": selected_level,
        "data_seed": int(data_seed),
        "coarsen_train": bool(coarsen_train),
        "gaps": list(profile.gaps),
        "probabilities": list(profile.probabilities),
        "split_ranges": {name: list(map(int, bounds)) for name, bounds in split_ranges.items()},
        "splits": manifests,
    }
    for split, (start, stop) in split_ranges.items():
        manifest["splits"][split]["forecast_origins"] = list(range(
            int(start) + int(observation_length), int(stop) - int(forecast_horizon) + 1))
    return (view, manifest)


def write_japan_irregular_manifest(path: str | Path, manifest: Mapping) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
