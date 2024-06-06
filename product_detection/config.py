from dataclasses import dataclass
from typing import Literal

KNOWN_CLASSES = [
    'Баранина', 
    'Говядина', 
    'Индейка', 
    'Кура', 
    'Свинина', 
    'Цыпленок'
]

@dataclass
class Scheduler:
    name: str

@dataclass
class Scheduler_ReduceOnPlateau(Scheduler):
    patience: int
    factor: float

@dataclass
class Scheduler_OneCycleLR(Scheduler):
    expand_lr: int

@dataclass
class Dataset:
    path_to_data: str
    vocab_size: int
    path_to_bpe_dir: str
    output_dir: str
    chunk_lenght: int
    pad_value: int
    need_to_train_bpe: bool
    test_size_split: float

@dataclass
class Training:
    project_name: str
    train_name: str
    seed: int
    epochs: int
    batch: int
    lr: float
    wandb_path: str
    model_path: str
    save_best_of: int
    checkpoint_monitor: str
    early_stopping_patience: int

@dataclass
class Model:
    embedding_dim: int
    layers: int
    heads: int
    mlp_dim: int
    qkv_bias: bool
    dropout: float
    norm_type: Literal["postnorm", "prenorm"]
    num_output_classes: int

@dataclass
class Params:
    dataset: Dataset
    training: Training
    model: Model
    scheduler: Scheduler