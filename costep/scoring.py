import numpy as np
import torch

from .intervals import interval_event_tensors, summarize_interval_events


def report_error_metrics(prediction, target, reporting_metadata):
    if reporting_metadata is None:
        raise ValueError("Scoring requires the saved reporting calendar")
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim == 3:
        prediction = prediction[..., None]
        target = target[..., None]
    metadata = {key: torch.as_tensor(value) for key, value in reporting_metadata.items()}
    events = interval_event_tensors(
        torch.as_tensor(prediction).float(), torch.as_tensor(target).float(), metadata
    )
    summary = summarize_interval_events(events)
    if not summary["event_count"]:
        raise ValueError("No complete reporting events in evaluation support")
    return summary
