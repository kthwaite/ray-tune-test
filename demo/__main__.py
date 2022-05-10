import toml
from hyperopt import hp
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from torchvision import datasets
from typing import Any, Dict

from .config import Config, TrainConfig
from .train import train_mnist


def _wrap_train_mnist(raw_cfg: Dict[str, Any]):
    cfg = TrainConfig.parse_obj(raw_cfg)
    train_mnist(cfg)


def hy_search():
    cfg = Config()
    datasets.MNIST(cfg.train.data_dir, train=True, download=True)

    raw = cfg.train.dict()

    raw["lr"] = hp.loguniform("lr", 1e-10, 0.1)
    raw["momentum"] = hp.uniform("momentum", 0.1, 0.9)

    hyperopt_search = HyperOptSearch(raw, metric="mean_accuracy", mode="max")

    analysis = tune.run(
        _wrap_train_mnist,
        num_samples=cfg.tune.num_samples,
        search_alg=hyperopt_search,
        local_dir=cfg.tune.ray_dir,
    )

    best = analysis.get_best_config("mean_accuracy", mode="max")
    print(best)
    cfg.train = TrainConfig.parse_obj(best)
    with open("best_config.toml", "w") as f:
        toml.dump(cfg.dict(), f)


if __name__ == "__main__":
    hy_search()
