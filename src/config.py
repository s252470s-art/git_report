from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 5
