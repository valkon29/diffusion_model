from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    dataset_path: str = "extracted_images"
    image_size: int = 64
    center_crop: bool = True
    random_flip: bool = True

    color_jitter: bool = True
    color_jitter_strength: float = 0.2
    random_rotation: int = 0

    model_config: dict = field(default_factory=lambda: {
        "sample_size": 64,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 1,
        "block_out_channels": (64, 128, 128, 256),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    })

    noise_scheduler_config: dict = field(default_factory=lambda: {
        "num_train_timesteps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
    })

    batch_size: int = 16
    num_epochs: int = 500
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-6
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    seed: int = 42

    output_dir: str = "output"
    checkpointing_steps: int = 500
    validation_epochs: int = 5
    num_validation_images: int = 8
    logging_dir: str = "logs"

    ddpm_num_steps: List[int] = field(default_factory=lambda: [1000])
    ddim_num_steps: List[int] = field(default_factory=lambda: [50, 100, 200])
    ddim_eta: float = 0.0


config = TrainingConfig()
