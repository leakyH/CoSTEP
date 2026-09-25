from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    dataset: str = "japan"
    ir: bool | None = None
    curve: bool = True
    consistency: bool = True
    post: bool = True
    capc_weight: float | None = None
    us_split: str = "811"
    curriculum: str = "none"
    level: str = "L0"
    seed: int = 42
    reporting_seed: int = 42
    batch_size: int = 32
    epochs: int = 500
    patience: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-8
    post_epochs: int = 500
    post_patience: int = 50
    post_learning_rate: float = 5e-4
    post_start_epoch: int = 100
    device: str = "cpu"
    output: str = "outputs/run"
    threads: int = 4
    sir_decoder_variant: str = "delay"
    mepognn_ablation: bool = False

    def __post_init__(self):
        if self.ir is None:
            object.__setattr__(self, "ir", self.dataset != "japan")
        if self.capc_weight is None:
            object.__setattr__(self, "capc_weight", 1000.0 if self.dataset == "us" else 10.0)

    def validate(self):
        if self.mepognn_ablation:
            expected_decoder = "delay" if self.dataset == "us" else "dynamic"
            if (self.dataset not in {"japan", "us"} or self.curve
                    or self.ir != (self.dataset == "us")
                    or self.sir_decoder_variant != expected_decoder):
                raise ValueError("MepoGNN ablation requires the bundled dataset-specific configuration")
        if self.sir_decoder_variant not in {"delay", "dynamic"}:
            raise ValueError("sir_decoder_variant must be delay or dynamic")
        if self.sir_decoder_variant == "dynamic" and not self.mepognn_ablation and (
            self.dataset != "japan" or self.ir or not self.curve
        ):
            raise ValueError("dynamic decoder requires JAPAN-Regular CoSTEP with curves")
        if self.dataset not in {"japan", "us", "japan_irregular"}:
            raise ValueError("dataset must be japan, us or japan_irregular")
        if self.us_split != "811":
            raise ValueError("us_split must be 811")
        if self.curriculum not in {"fixed_epoch", "none"}:
            raise ValueError("Invalid curriculum")
        if self.level not in {"L0", "L1", "L2", "L3", "L4"}:
            raise ValueError("level must be L0–L4")
        if self.capc_weight < 0:
            raise ValueError("capc_weight must be nonnegative")
        if self.dataset == "japan_irregular" and self.capc_weight != 10:
            raise ValueError("JAPAN-Irregular uses CAPC weight 10")
        if self.dataset == "japan" and self.ir:
            raise ValueError("JAPAN-Regular uses CoSTEP")
        for key in ("batch_size", "epochs", "patience", "post_epochs", "post_patience", "threads"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.learning_rate <= 0 or self.post_learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.post_start_epoch < 0:
            raise ValueError("post_start_epoch must be nonnegative")
        return self

    def parameters(self):
        self.validate()
        irregular = self.dataset == "japan_irregular"
        native_dynamic = self.mepognn_ablation and self.dataset == "japan"
        return dict(
            GPU=self.device,
            output_dir=str(Path(self.output).resolve()),
            dataset="US_STATE" if self.dataset == "us" else "JAPAN",
            obs_len=14,
            pred_len=14,
            split_ratio=[8, 1, 1] if self.dataset == "us" else [6, 1, 1],
            split_mode="calendar_disjoint" if irregular else "ratio",
            non_test_overlap=not irregular,
            us_split_variant=self.us_split,
            us_protocol="synthetic750" if self.dataset == "us" else None,
            reporting_mode="interval" if self.ir else "daily",
            stock_estimator="causal_interval_anchor" if self.ir else "legacy_daily",
            loss="report_mae" if self.ir else "MAE",
            selection_metric="report_mae" if self.dataset != "japan" else "legacy_mae",
            gamma_floor_mode="observation_stock",
            gamma_floor_straight_through=True,
            japan_irregular=irregular,
            japan_irregular_level=self.level,
            reporting_data_seed=self.reporting_seed,
            reporting_augmentation="coarsened" if irregular else "native",
            emit_reporting_metrics=True,
            random_seed=self.seed,
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            learn_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            curve=self.curve,
            sir_decoder_variant=self.sir_decoder_variant,
            mepognn_ablation=self.mepognn_ablation,
            consistency_enabled=self.consistency,
            consistency_method="legacy" if native_dynamic else "capc_src",
            consistency_loss_lambda=self.capc_weight if self.consistency else 0.0,
            stitched_left_loss_weight=1.0 if self.consistency and not native_dynamic else 0.0,
            stitched_right_loss_weight=1.0 if self.consistency and not native_dynamic else 0.0,
            base_full_horizon_training=self.curriculum == "none",
            post_train_epochs=self.post_epochs,
            post_train_patience=self.post_patience,
            post_train_lr=self.post_learning_rate,
            post_train_checkpoint_start_epoch=self.post_start_epoch,
            debug=False,
            release_config=asdict(self),
        )
