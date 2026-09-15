"""Sample split-contained reporting calendars and build causal interval metadata."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

REPORTING_KEYS = (
    "report_count",
    "report_mask",
    "report_interval",
    "report_rate",
    "elapsed_since_report",
    "previous_report_index",
    "valid_report_event",
    "causal_report_rate",
)


def as_tnc(values: np.ndarray, *, name: str = "counts") -> np.ndarray:
    """Return values as ``[time, node, 1]`` without changing their dtype."""
    array = np.asarray(values)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim != 3 or array.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [T,N] or [T,N,1], got {array.shape}")
    return array


@dataclass(frozen=True)
class FrozenGapProfile:
    gaps: tuple[int, ...]
    probabilities: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: Mapping) -> "FrozenGapProfile":
        gaps = tuple((int(value) for value in payload["gaps"]))
        probabilities = tuple((float(value) for value in payload["probabilities"]))
        if not gaps or len(gaps) != len(probabilities):
            raise ValueError("gap profile needs equally sized non-empty gaps/probabilities")
        if any((value < 1 for value in gaps)) or len(set(gaps)) != len(gaps):
            raise ValueError("gap values must be unique positive integers")
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if np.any(probability_array < 0) or not np.isclose(probability_array.sum(), 1.0):
            raise ValueError("gap probabilities must be non-negative and sum to one")
        return cls(gaps, probabilities)


def _draw_block_lengths(
    length: int, profile: FrozenGapProfile, rng: np.random.Generator
) -> list[int]:
    blocks = []
    remaining = int(length)
    gaps = np.asarray(profile.gaps, dtype=np.int64)
    probabilities = np.asarray(profile.probabilities, dtype=np.float64)
    while remaining > 0:
        sampled = int(rng.choice(gaps, p=probabilities))
        block_length = min(sampled, remaining)
        blocks.append(block_length)
        remaining -= block_length
    return blocks


def coarsen_reporting_calendar(
    counts: np.ndarray, *, start: int, stop: int, profile: FrozenGapProfile, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Move case mass to sampled report dates within one half-open range."""
    original = as_tnc(counts)
    if np.any(original < 0):
        raise ValueError("reporting coarsening requires non-negative counts")
    start = int(start)
    stop = int(stop)
    if not 0 <= start < stop <= original.shape[0]:
        raise ValueError(f"invalid augmentation range [{start}, {stop}) for T={original.shape[0]}")
    output = np.array(original, copy=True)
    schedule = original > 0
    output[start:stop] = 0
    schedule[start:stop] = False
    node_summaries = []
    for node in range(original.shape[1]):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(node)]))
        cursor = start
        blocks = _draw_block_lengths(stop - start, profile, rng)
        for block_length in blocks:
            end = cursor + int(block_length)
            output[end - 1, node, 0] = original[cursor:end, node, 0].sum(dtype=np.float64)
            schedule[end - 1, node, 0] = True
            cursor = end
        if cursor != stop:
            raise AssertionError(f"node {node} schedule ended at {cursor}, expected {stop}")
        node_summaries.append(
            {"node_index": node, "block_count": len(blocks), "block_lengths": blocks}
        )
    before = original[start:stop].sum(axis=0, dtype=np.float64)
    after = output[start:stop].sum(axis=0, dtype=np.float64)
    if not np.allclose(before, after, rtol=0, atol=1e-06):
        raise AssertionError("reporting augmentation failed per-node conservation")
    if np.any((output[start:stop] > 0) & ~schedule[start:stop]):
        raise AssertionError("coarsened positive counts exist outside generated report dates")
    manifest = {
        "schema": "costep.reporting_calendar.v1",
        "seed": int(seed),
        "start": start,
        "stop": stop,
        "gaps": list(profile.gaps),
        "probabilities": list(profile.probabilities),
        "total_before": float(before.sum()),
        "total_after": float(after.sum()),
        "per_node_conserved": True,
        "node_summaries": node_summaries,
    }
    return (output.astype(original.dtype, copy=False), schedule, manifest)


def build_reporting_metadata(
    report_counts: np.ndarray,
    *,
    report_mask: np.ndarray | None = None,
    reset_indices: tuple[int, ...] | list[int] = (),
) -> dict[str, np.ndarray]:
    """Build global, causal interval metadata on the original daily grid."""
    counts = as_tnc(report_counts, name="report counts").astype(np.float64, copy=False)
    negative = np.argwhere(counts < 0)
    if negative.size:
        examples = [
            {"time_index": int(t), "node_index": int(n), "value": float(counts[t, n, c])}
            for (t, n, c) in negative[:20]
        ]
        raise ValueError(
            f"interval reporting requires nonnegative counts; found {len(negative)} negative values, examples={examples}"
        )
    if report_mask is None:
        mask = counts > 0
    else:
        mask = as_tnc(np.asarray(report_mask), name="report_mask").astype(bool, copy=False)
        if mask.shape != counts.shape:
            raise ValueError(f"report_mask shape {mask.shape} != counts shape {counts.shape}")
        if np.any((counts > 0) & ~mask):
            raise ValueError("positive report counts must be marked in report_mask")
    reset_set = {int(value) for value in reset_indices}
    if any((value < 0 or value >= counts.shape[0] for value in reset_set)):
        raise ValueError(f"reset_indices outside calendar [0, {counts.shape[0]})")
    time_steps, nodes, _ = counts.shape
    intervals = np.zeros_like(counts, dtype=np.int64)
    rates = np.zeros_like(counts, dtype=np.float64)
    elapsed = np.zeros_like(counts, dtype=np.int64)
    previous = np.full_like(counts, -1, dtype=np.int64)
    valid = np.zeros_like(mask, dtype=bool)
    causal_rate = np.zeros_like(counts, dtype=np.float64)
    for node in range(nodes):
        last_report = -1
        segment_start = 0
        last_rate = 0.0
        for time_index in range(time_steps):
            if time_index in reset_set:
                segment_start = time_index
                last_report = -1
                last_rate = 0.0
            previous[time_index, node, 0] = last_report
            if last_report >= 0:
                elapsed[time_index, node, 0] = time_index - last_report
            if mask[time_index, node, 0]:
                if last_report >= 0:
                    interval = time_index - last_report
                    valid[time_index, node, 0] = True
                else:
                    interval = time_index - segment_start + 1
                rate = counts[time_index, node, 0] / float(interval)
                intervals[time_index, node, 0] = interval
                rates[time_index, node, 0] = rate
                elapsed[time_index, node, 0] = interval
                last_report = time_index
                last_rate = rate
            causal_rate[time_index, node, 0] = last_rate
    return {
        "report_count": counts.astype(np.float32),
        "report_mask": mask,
        "report_interval": intervals,
        "report_rate": rates.astype(np.float32),
        "elapsed_since_report": elapsed,
        "previous_report_index": previous,
        "valid_report_event": valid,
        "causal_report_rate": causal_rate.astype(np.float32),
    }


def build_history_reporting_payload(
    metadata: Mapping[str, np.ndarray], *, origin_indices: np.ndarray, obs_len: int
) -> dict[str, np.ndarray]:
    counts = as_tnc(metadata["report_count"], name="report_count")
    origins = np.asarray(origin_indices, dtype=np.int64).reshape(-1)
    history_len = int(obs_len)
    if history_len <= 0:
        raise ValueError("obs_len must be positive")
    if np.any(origins < history_len) or np.any(origins > counts.shape[0]):
        raise ValueError(
            f"forecast origins must have a complete history within [{history_len}, {counts.shape[0]}], got {origins.tolist()}"
        )
    history_keys = (
        "report_mask",
        "report_interval",
        "report_rate",
        "elapsed_since_report",
        "causal_report_rate",
    )
    return {
        f"history_{key}": np.asarray(
            [metadata[key][origin - history_len : origin] for origin in origins]
        )
        for key in history_keys
    }


def build_future_reporting_payload(
    metadata: Mapping[str, np.ndarray], *, origin_indices: np.ndarray, pred_len: int
) -> dict[str, np.ndarray]:
    counts = as_tnc(metadata["report_count"], name="report_count")
    origins = np.asarray(origin_indices, dtype=np.int64).reshape(-1)
    horizon = int(pred_len)
    if horizon <= 0:
        raise ValueError("pred_len must be positive")
    if np.any(origins < 0) or np.any(origins >= counts.shape[0]):
        raise ValueError(
            f"forecast origins must lie in [0, {counts.shape[0]}), got {origins.tolist()}"
        )
    payload = {
        "future_report_mask": [],
        "future_report_interval": [],
        "future_report_rate": [],
        "future_interval_start": [],
        "future_event_valid": [],
        "origin_index": origins,
    }
    nodes = counts.shape[1]
    for origin in origins.tolist():
        available = min(horizon, counts.shape[0] - origin)
        shape = (horizon, nodes, 1)
        future_mask = np.zeros(shape, dtype=metadata["report_mask"].dtype)
        future_interval = np.zeros(shape, dtype=metadata["report_interval"].dtype)
        future_rate = np.zeros(shape, dtype=metadata["report_rate"].dtype)
        previous = np.full(shape, -1, dtype=metadata["previous_report_index"].dtype)
        globally_valid = np.zeros(shape, dtype=metadata["valid_report_event"].dtype)
        stop = origin + available
        future_mask[:available] = metadata["report_mask"][origin:stop]
        future_interval[:available] = metadata["report_interval"][origin:stop]
        future_rate[:available] = metadata["report_rate"][origin:stop]
        previous[:available] = metadata["previous_report_index"][origin:stop]
        globally_valid[:available] = metadata["valid_report_event"][origin:stop]
        interval_start = previous + 1 - origin
        valid = future_mask & globally_valid & (interval_start >= 0)
        payload["future_report_mask"].append(future_mask)
        payload["future_report_interval"].append(future_interval)
        payload["future_report_rate"].append(future_rate)
        payload["future_interval_start"].append(np.where(valid, interval_start, -1))
        payload["future_event_valid"].append(valid)
    return {
        key: value if key == "origin_index" else np.asarray(value)
        for (key, value) in payload.items()
    }


def build_window_reporting_payload(
    metadata: Mapping[str, np.ndarray], *, obs_len: int, pred_len: int
) -> dict[str, np.ndarray]:
    counts = as_tnc(metadata["report_count"], name="report_count")
    sample_count = counts.shape[0] - int(obs_len) - int(pred_len) + 1
    if sample_count <= 0:
        raise ValueError("reporting series is shorter than obs_len + pred_len")
    origins = np.arange(sample_count, dtype=np.int64) + int(obs_len)
    result = build_history_reporting_payload(metadata, origin_indices=origins, obs_len=obs_len)
    result.update(
        build_future_reporting_payload(metadata, origin_indices=origins, pred_len=pred_len)
    )
    return result
