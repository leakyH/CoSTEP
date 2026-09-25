import json
from pathlib import Path

import numpy as np

from .scoring import report_error_metrics


def score(prediction, target, metadata):
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and match target shape")
    valid = np.isfinite(target)
    if not valid.any():
        raise ValueError("No finite targets")
    error = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    total = float(np.abs(target[valid].astype(np.float64)).sum())
    result = dict(
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        wape=float(100.0 * np.abs(error).sum() / total) if total > 1e-12 else None,
        elements=int(valid.sum()),
    )
    for day in (3, 7, 14):
        if prediction.shape[1] >= day:
            horizon_error = prediction[:, day - 1].astype(np.float64) - target[:, day - 1].astype(
                np.float64
            )
            result[f"h{day}_mae"] = float(np.abs(horizon_error[np.isfinite(horizon_error)]).mean())
    event_metadata = {k: v for k, v in metadata.items()
                      if k.startswith("future_") or k == "origin_index"}
    report = report_error_metrics(prediction, target, reporting_metadata=event_metadata)
    result.update(report)
    return result


def save_evaluation(trainer, loader, checkpoint, output, stage):
    prediction, target, metadata = trainer.predict(
        loader, checkpoint, apply_adaptor=stage == "post"
    )
    dataset = trainer.params["release_config"]["dataset"]
    # Scoring is task-dependent, including for the daily-input CoSTEP control.
    context = {"dataset": dataset, "reporting_mode": "daily" if dataset == "japan" else "interval"}
    metadata["dataset_context"] = np.asarray(json.dumps(context, sort_keys=True))
    if "node_identity_provenance" in trainer.raw_data:
        metadata["node_identity_provenance"] = np.asarray(json.dumps(
            trainer.raw_data["node_identity_provenance"], sort_keys=True))
    for key in ("node_ids", "node_names", "dates"):
        if key in trainer.raw_data:
            values = np.asarray(trainer.raw_data[key])
            metadata[key] = values.astype(str) if key == "dates" else values
    return save_predictions(prediction, target, metadata, output, stage)


def save_predictions(prediction, target, metadata, output, prefix):
    if "dataset_context" not in metadata or "origin_index" not in metadata:
        raise ValueError("Prediction export requires dataset_context and origin_index for reproducibility")
    path = Path(output)
    artifact = path / (prefix + "_predictions.npz")
    if artifact.exists():
        raise FileExistsError(artifact)
    metrics = score(prediction, target, metadata)
    np.savez_compressed(artifact, prediction=prediction, target=target, **metadata)
    (path / (prefix + "_metrics.json")).write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    return metrics


def rescore(path):
    with np.load(path, allow_pickle=False) as arrays:
        if "dataset_context" not in arrays or "origin_index" not in arrays:
            raise ValueError("Scoring requires saved dataset_context and origin_index; no implicit daily/interval choice")
        return score(
            arrays["prediction"],
            arrays["target"],
            {k: arrays[k] for k in arrays.files if k not in {"prediction", "target"}},
        )
