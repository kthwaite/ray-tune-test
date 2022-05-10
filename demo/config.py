import os
from pydantic import BaseModel


class TrainConfig(BaseModel):
    epochs: int = 10
    momentum: float = 0
    lr: float = 0.01
    batch_size: int = 64
    checkpoint_dir: str = "./checkpoint"
    data_dir: str = "./data"

    def checkpoint_dir_exists(self) -> bool:
        return os.path.isdir(self.checkpoint_dir)

    def ensure_checkpoint_dir(self) -> bool:
        """Ensure the checkpoint dir exists, returning True if it was created, and
        False if it already existed.
        """
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=False)
            return True
        except OSError:
            return False


class TuneConfig(BaseModel):
    ray_dir: str = "./ray"
    num_samples: int = 10


class Config(BaseModel):
    tune: TuneConfig = TuneConfig()
    train: TrainConfig = TrainConfig()
