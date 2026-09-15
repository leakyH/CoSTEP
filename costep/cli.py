import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shlex
import sys

import numpy as np
import torch

from .config import Experiment, preset
from .data import WindowBuilder
from .datasets import load_dataset
from .evaluation import rescore, save_evaluation
from .training import Trainer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare(config, data_root=None):
    config.validate()
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    data, identity = load_dataset(config.dataset, data_root)
    params = config.parameters()
    params["data_identity"] = identity
    loaders = WindowBuilder(14, 14, params["split_ratio"]).get_data_loader(data, params)
    trainer = Trainer(params, data, max_od=loaders["max_od"])
    return trainer, loaders


def run(config, action="train", checkpoint=None, data_root=None):
    config.validate()
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if action in {"adapt", "evaluate"} and checkpoint is None:
        raise ValueError(f"{action} requires a checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    status_path = output / "status.json"
    status_path.write_text(json.dumps({"state": "RUNNING", "action": action}))
    try:
        trainer, loaders = prepare(config, data_root)
        restricted = {k: v for k, v in loaders.items() if k != "test"}
        if action == "train":
            trainer.train(
                restricted,
                ["train", "validate"],
                early_stop_patience=config.patience,
                checkpoint_path=trainer.get_checkpoint_path(stage="base"),
            )
            base = trainer.get_checkpoint_path(stage="base")
            if not Path(base).is_file():
                raise RuntimeError(
                    "No eligible base checkpoint: the curriculum may need more epochs"
                )
            if config.post:
                trainer.run_post_training(restricted, base)
            # Use a fresh base model after Post has added adaptor parameters.
            base_trainer = Trainer(trainer.params, trainer.raw_data, max_od=trainer.max_od)
            results = {"base": save_evaluation(base_trainer, loaders["test"], base, output, "base")}
            if config.post:
                results["post"] = save_evaluation(
                    trainer,
                    loaders["test"],
                    trainer.get_checkpoint_path(stage="post"),
                    output,
                    "post",
                )
        elif action == "adapt":
            trainer.run_post_training(restricted, str(checkpoint))
            results = {
                "post": save_evaluation(
                    trainer,
                    loaders["test"],
                    trainer.get_checkpoint_path(stage="post"),
                    output,
                    "post",
                )
            }
        elif action == "evaluate":
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            stage = loaded["stage"]
            results = {
                stage: save_evaluation(trainer, loaders["test"], str(checkpoint), output, stage)
            }
        else:
            raise ValueError(f"Unknown action: {action}")
        status_path.write_text(
            json.dumps(
                {"state": "COMPLETE", "action": action, "results": results},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        return results
    except BaseException as exc:
        status_path.write_text(json.dumps({"state": "FAILED", "error": str(exc)}, indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-experiment CoSTEP / CoSTEP-IR training and evaluation"
    )
    parser.add_argument("action", choices=["train", "adapt", "evaluate", "score"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=["japan", "us", "japan_irregular"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--curve", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ir", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--consistency", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--us-split", choices=["811"])
    parser.add_argument("--curriculum", choices=["fixed_epoch", "none"])
    parser.add_argument("--level", choices=["L0", "L1", "L2", "L3", "L4"])
    for name in [
        "seed",
        "reporting-seed",
        "batch-size",
        "epochs",
        "patience",
        "post-epochs",
        "post-patience",
        "post-start-epoch",
        "threads",
    ]:
        parser.add_argument("--" + name, type=int)
    for name in ["capc-weight", "learning-rate", "post-learning-rate", "weight-decay"]:
        parser.add_argument("--" + name, type=float)
    args = parser.parse_args(argv)
    if args.action == "score":
        if args.predictions is None:
            parser.error("score requires --predictions")
        print(json.dumps(rescore(args.predictions), indent=2, allow_nan=False))
        return
    if args.action in {"adapt", "evaluate"} and args.checkpoint is None:
        parser.error("adapt/evaluate require --checkpoint")
    if args.config is not None:
        values = json.loads(args.config.read_text())
    elif args.checkpoint is not None:
        values = dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"])
    else:
        values = asdict(preset(args.dataset or "japan"))
    for key in Experiment.__dataclass_fields__:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    if args.output is None and args.action in {"adapt", "evaluate"}:
        parser.error("adapt/evaluate require a new --output directory")
    print(
        json.dumps(
            run(Experiment(**values), args.action, args.checkpoint, args.data_root),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
