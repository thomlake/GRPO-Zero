from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = 'Qwen/Qwen2.5-3B-Instruct'


class DataConfig(BaseModel):
    path: str = 'Countdown-Tasks-3to4'
    test_size: int = 128


class TrainingConfig(BaseModel):
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    random_seed: int = 1337
    max_prompt_len: int = 256
    max_new_tokens: int = 1024
    batch_size: int = 256
    num_questions_per_batch: int = 32
    # Number of examples per gradient accumulation step
    micro_batch_size: int = 2
    max_grad_norm: float = 1.0
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    ckpt_dir: str = "ckpt"
    log_dir: str = "logs"
    skip_unfinished_episodes: bool = False
    ckpt_save_interval: int = 100
    eval_interval: int = 10
    memory_efficient_adamw: bool = False


class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def load_yaml(cls, file: str | Path) -> Self:
        with open(file) as fp:
            data = yaml.safe_load(fp)

        return cls.model_validate(data, strict=True)
