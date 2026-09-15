import json
from pathlib import Path


def record_epoch(trainer, data_loader, epoch, epoch_offset):
    row = dict(
        stage=trainer.training_stage,
        epoch=int(epoch),
        checkpoint_epoch=int(epoch + epoch_offset),
        losses=dict(trainer.last_epoch_mode_losses),
        curriculum=trainer.curriculum_horizon_trace[-1],
        finite_gradient_steps=getattr(trainer, "_finite_gradient_steps", 0),
        max_gradient_norm=getattr(trainer, "_max_gradient_norm", 0.0),
    )
    with (Path(trainer.params["output_dir"]) / "epochs.jsonl").open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
