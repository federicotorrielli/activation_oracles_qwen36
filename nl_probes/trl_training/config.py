from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict, Field
from trl import SFTConfig


class EvalConfig(BaseModel, extra="forbid"):
    random_seed: int = 42

    model_name: str = Field(...)

    verbose: bool = True

    model_lora_dir: str = "lora_models"
    wandb_info: str = ""
    wandb_project: str = "trl_demo"

    # ------------- convenience IO helpers -------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)  # full type check

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump()))


class FrozenEvalConfig(EvalConfig):
    model_config = ConfigDict(frozen=True)


def CustomSFTConfig(
    model_name: str,
    batch_size: int = 8,
    real_batch_size: int = 16,
    **overrides: Any,
) -> SFTConfig:
    """
    Factory returning an SFTConfig with repo defaults.

    Keeping this as a function avoids breaking dataclasses.replace inside TRL.
    """
    cfg = dict(
        packing=False,
        max_length=1024,
        num_train_epochs=1.0,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, real_batch_size // batch_size),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        # optim="paged_adamw_8bit",
        per_device_eval_batch_size=batch_size * 2,
        weight_decay=0.01,
        learning_rate=5e-5,
        # lr_scheduler_type="linear",
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
        bf16=True,
        eval_strategy="steps",
        eval_steps=50,
        eval_on_start=True,
        save_steps=50,  # match eval_steps so best step is saved
        load_best_model_at_end=True,  # needed for early stopping + restore best
        metric_for_best_model="eval_loss",  # default metric, uses eval loss
        greater_is_better=False,  # lower eval_loss is better
        save_total_limit=2,  # keep disk usage sane
        # max_steps=None,
        output_dir="sft_outputs",
        logging_steps=1,
        run_name=model_name,
        # report_to="wandb",
        report_to="none",
        # completion_only_loss=True,
        # assistant_only_loss=True,
        seed=42,
    )
    cfg.update(overrides)
    return SFTConfig(**cfg)


class CustomLoraConfig(LoraConfig):
    def __init__(self):
        super().__init__(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
