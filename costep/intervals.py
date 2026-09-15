"""Reporting metadata and reportMAE on the full daily calendar.

Explicit masks identify report events, including genuine zero-count reports.
When no mask is supplied, positive counts identify the native report events.
For complete forecast intervals, reportMAE compares summed predictions with
reported counts, divides absolute error by interval length, and averages events.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Mapping
import numpy as np
import torch

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
REPORT_MAE_SELECTION_METRICS = frozenset({"report_mae", "interval_normalized_mae"})


def is_report_mae_selection_metric(value: str | None) -> bool:
    """Return whether a CLI metric name selects reportMAE."""
    return value in REPORT_MAE_SELECTION_METRICS


def _as_tnc(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim != 3 or array.shape[-1] != 1:
        raise ValueError(f"report counts must have shape [T,N] or [T,N,1], got {array.shape}")
    return array


from .reporting import (
    build_window_reporting_payload,
    build_reporting_metadata,
    build_history_reporting_payload,
    build_future_reporting_payload,
)


def interval_event_tensors(
    daily_prediction: torch.Tensor, daily_target: torch.Tensor, metadata: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Gather predicted and observed counts for complete future report events."""
    if daily_prediction.shape != daily_target.shape:
        raise ValueError(
            f"prediction/target shapes differ: {daily_prediction.shape} vs {daily_target.shape}"
        )
    if daily_prediction.dim() != 4 or daily_prediction.shape[-1] != 1:
        raise ValueError(f"expected [B,T,N,1], got {tuple(daily_prediction.shape)}")
    valid = metadata["future_event_valid"].to(device=daily_prediction.device, dtype=torch.bool)
    starts = metadata["future_interval_start"].to(device=daily_prediction.device, dtype=torch.long)
    intervals = metadata["future_report_interval"].to(
        device=daily_prediction.device, dtype=daily_prediction.dtype
    )
    if valid.shape != daily_prediction.shape:
        raise ValueError(
            f"future_event_valid shape {valid.shape} != prediction shape {daily_prediction.shape}"
        )
    target_valid_mask = metadata.get("target_valid_mask")
    if target_valid_mask is not None:
        target_valid_mask = target_valid_mask.to(device=daily_prediction.device, dtype=torch.bool)
        try:
            target_valid_mask = torch.broadcast_to(target_valid_mask, daily_target.shape)
        except RuntimeError as error:
            raise ValueError(
                f"target_valid_mask cannot broadcast to target shape: {target_valid_mask.shape} vs {daily_target.shape}"
            ) from error
        valid = valid & target_valid_mask
    valid = valid & torch.isfinite(daily_target)
    cumulative = torch.cumsum(daily_prediction, dim=1)
    batch_size, horizon, nodes, _ = daily_prediction.shape
    end_index = torch.arange(horizon, device=daily_prediction.device).view(1, horizon, 1, 1)
    end_index = end_index.expand(batch_size, horizon, nodes, 1)
    batch_index = torch.arange(batch_size, device=daily_prediction.device).view(batch_size, 1, 1, 1)
    batch_index = batch_index.expand(batch_size, horizon, nodes, 1)
    node_index = torch.arange(nodes, device=daily_prediction.device).view(1, 1, nodes, 1)
    node_index = node_index.expand(batch_size, horizon, nodes, 1)
    previous_index = (starts - 1).clamp_min(0)
    previous_cumulative = torch.gather(cumulative, 1, previous_index)
    predicted_counts = torch.gather(cumulative, 1, end_index)
    predicted_counts = torch.where(
        starts > 0, predicted_counts - previous_cumulative, predicted_counts
    )
    observed_counts = daily_target
    selected = valid
    return {
        "predicted_count": predicted_counts[selected],
        "observed_count": observed_counts[selected],
        "interval": intervals[selected].clamp_min(1.0),
        "valid_mask": valid,
        "end_offset": end_index[selected] + 1,
        "batch_index": batch_index[selected],
        "node_index": node_index[selected],
    }


def complete_horizon_cumulative_mae(
    daily_prediction: torch.Tensor, daily_target: torch.Tensor
) -> float | None:
    """Return cumulative MAE using only origins with a complete target horizon.

    Partial-label protocols intentionally contain NaNs after each origin's
    legal label boundary.  Summing those origins would turn an otherwise valid
    reportMAE artifact into NaN, so the cumulative metric is undefined when no
    complete origin remains.
    """
    if daily_prediction.shape != daily_target.shape:
        raise ValueError(
            f"prediction/target shapes differ: {daily_prediction.shape} vs {daily_target.shape}"
        )
    if daily_prediction.dim() != 4 or daily_prediction.shape[-1] != 1:
        raise ValueError(f"expected [B,T,N,1], got {tuple(daily_prediction.shape)}")
    complete_origins = torch.isfinite(daily_target).reshape(daily_target.shape[0], -1).all(dim=1)
    if not bool(complete_origins.any().item()):
        return None
    cumulative_error = torch.abs(
        daily_prediction[complete_origins].sum(dim=1) - daily_target[complete_origins].sum(dim=1)
    )
    return float(cumulative_error.mean().item())


def interval_normalized_mae_loss(
    daily_prediction: torch.Tensor, daily_target: torch.Tensor, metadata: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    events = interval_event_tensors(daily_prediction, daily_target, metadata)
    if events["predicted_count"].numel() == 0:
        return (None, events)
    errors = torch.abs(events["predicted_count"] - events["observed_count"])
    return ((errors / events["interval"]).mean(), events)


def report_mae_loss(
    daily_prediction: torch.Tensor, daily_target: torch.Tensor, metadata: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Compute reportMAE; alias of the historical interval-normalized loss."""
    return interval_normalized_mae_loss(daily_prediction, daily_target, metadata)


def summarize_interval_events(events: Mapping[str, torch.Tensor]) -> dict[str, float]:
    predicted = events["predicted_count"].detach().float().cpu()
    observed = events["observed_count"].detach().float().cpu()
    interval = events["interval"].detach().float().cpu()
    if predicted.numel() == 0:
        return {"event_count": 0}
    pred_rate = predicted / interval
    obs_rate = observed / interval
    diff = pred_rate - obs_rate
    count_diff = predicted - observed
    denom = torch.sum(torch.abs(obs_rate)).clamp_min(1e-08)
    return {
        "event_count": int(predicted.numel()),
        "report_mae": float(torch.mean(torch.abs(diff))),
        "interval_normalized_mae": float(torch.mean(torch.abs(diff))),
        "interval_normalized_rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "interval_normalized_mape": float(
            torch.mean(torch.abs(diff) / torch.abs(obs_rate).clamp_min(1e-08)) * 100.0
        ),
        "interval_normalized_rae": float(torch.sum(torch.abs(diff)) / denom),
        "interval_count_mae": float(torch.mean(torch.abs(count_diff))),
        "interval_count_rmse": float(torch.sqrt(torch.mean(count_diff.square()))),
        "report_count_mae": float(torch.mean(torch.abs(count_diff))),
        "report_count_rmse": float(torch.sqrt(torch.mean(count_diff.square()))),
    }


def build_reporting_audit(
    raw_data: Mapping[str, np.ndarray],
    metadata: Mapping[str, np.ndarray],
    *,
    protocol: str | None,
    mode_indices: Mapping[str, np.ndarray] | None = None,
    mode_len: Mapping[str, int] | None = None,
    obs_len: int = 14,
    pred_len: int = 14,
) -> tuple[dict, list[dict]]:
    counts = _as_tnc(metadata["report_count"])
    positive = counts > 0
    fractional = np.abs(counts - np.round(counts)) > 1e-08
    gap_values = metadata["report_interval"][metadata["valid_report_event"]]
    summary = {
        "protocol": protocol,
        "time_steps": int(counts.shape[0]),
        "nodes": int(counts.shape[1]),
        "negative_count": int(np.sum(counts < 0)),
        "zero_count": int(np.sum(counts == 0)),
        "positive_count": int(np.sum(positive)),
        "fractional_count": int(np.sum(fractional)),
        "positive_fraction": float(np.mean(positive)),
        "max_report_gap": int(gap_values.max()) if gap_values.size else 0,
        "report_total": float(counts.sum()),
        "positive_event_total": float(counts[positive].sum()),
        "interval_event_total": float(counts[metadata["valid_report_event"]].sum()),
        "excluded_first_event_total": float(
            counts[positive & ~metadata["valid_report_event"]].sum()
        ),
        "aligned_lengths": {
            key: int(np.asarray(raw_data[key]).shape[0])
            for key in ("y", "SIR", "node", "od")
            if key in raw_data
        },
        "synthetic_od": protocol in {"synthetic750", "exploratory938"},
    }
    if "dates" in raw_data and len(raw_data["dates"]):
        dates = np.asarray(raw_data["dates"])
        summary["date_start"] = str(dates[0])
        summary["date_end"] = str(dates[-1])
        summary["date_count"] = int(len(dates))
    if "synthetic_od_start_index" in raw_data:
        boundary = int(raw_data["synthetic_od_start_index"])
        summary["synthetic_od_start_index"] = boundary
        if "dates" in raw_data and boundary < len(raw_data["dates"]):
            summary["synthetic_od_start_date"] = str(np.asarray(raw_data["dates"])[boundary])
    lengths = set(summary["aligned_lengths"].values())
    summary["date_aligned"] = len(lengths) <= 1
    summary["interval_total_conserved"] = bool(
        np.isclose(summary["report_total"], summary["positive_event_total"])
    )
    rows = []
    for node in range(counts.shape[1]):
        node_gaps = metadata["report_interval"][:, node, 0][
            metadata["valid_report_event"][:, node, 0]
        ]
        rows.append(
            {
                "node_index": node,
                "zero_fraction": float(np.mean(counts[:, node, 0] == 0)),
                "positive_events": int(np.sum(positive[:, node, 0])),
                "max_gap": int(node_gaps.max()) if node_gaps.size else 0,
                "gap_1": int(np.sum(node_gaps == 1)),
                "gap_2_3": int(np.sum((node_gaps >= 2) & (node_gaps <= 3))),
                "gap_4_7": int(np.sum((node_gaps >= 4) & (node_gaps <= 7))),
                "gap_8_14": int(np.sum((node_gaps >= 8) & (node_gaps <= 14))),
                "gap_gt14": int(np.sum(node_gaps > 14)),
            }
        )
    if mode_len is not None:
        if mode_indices is None:
            cursor = 0
            split_indices = {}
            for mode in ("train", "validate", "test"):
                length = int(mode_len.get(mode, 0))
                split_indices[mode] = np.arange(cursor, cursor + length)
                cursor += length
        else:
            split_indices = mode_indices
        window_payload = build_window_reporting_payload(
            metadata, obs_len=obs_len, pred_len=pred_len
        )
        summary["valid_events_by_split"] = {}
        summary["valid_interval_statistics_by_split"] = {}
        for mode, indices in split_indices.items():
            split_window_indices = np.asarray(indices, dtype=np.int64)
            valid = window_payload["future_event_valid"][split_window_indices]
            future_reports = window_payload["future_report_mask"][split_window_indices]
            counts_per_window_node = valid.sum(axis=1).reshape(-1)
            valid_count = int(valid.sum())
            report_count = int(future_reports.sum())
            pair_count = int(counts_per_window_node.size)
            summary["valid_events_by_split"][mode] = valid_count
            summary["valid_interval_statistics_by_split"][mode] = {
                "prediction_windows": int(valid.shape[0]),
                "nodes": int(valid.shape[2]),
                "window_node_pairs": pair_count,
                "valid_intervals": valid_count,
                "future_report_events": report_count,
                "mean_valid_intervals_per_window_node": (
                    float(counts_per_window_node.mean()) if pair_count else 0.0
                ),
                "valid_day_density": float(valid_count / valid.size) if valid.size else 0.0,
                "usable_future_report_fraction": (
                    float(valid_count / report_count) if report_count else 0.0
                ),
                "zero_valid_window_node_fraction": (
                    float(np.mean(counts_per_window_node == 0)) if pair_count else 0.0
                ),
            }
    return (summary, rows)


def write_reporting_audit(output_dir: str | Path, summary: Mapping, rows: list[Mapping]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reporting_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0].keys()) if rows else ["node_index"]
    with (output / "reporting_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "REPORTING_KEYS",
    "REPORT_MAE_SELECTION_METRICS",
    "is_report_mae_selection_metric",
    "build_reporting_metadata",
    "build_window_reporting_payload",
    "interval_event_tensors",
    "complete_horizon_cumulative_mae",
    "interval_normalized_mae_loss",
    "report_mae_loss",
    "summarize_interval_events",
    "build_reporting_audit",
    "write_reporting_audit",
]
