import torch
import torch.nn.functional as F
from ray import tune
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig
from .model import Model


EPOCH_SIZE = 512
TEST_SIZE = 256


def train(model: Model, opt: SGD, train_loader: DataLoader, device: torch.device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        opt.step()


def test(model: Model, test_loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_mnist(cfg: TrainConfig):
    # Data Setup
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = DataLoader(
        datasets.MNIST(
            cfg.data_dir, train=True, download=True, transform=mnist_transforms
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(cfg.data_dir, train=False, transform=mnist_transforms),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    model.to(device)

    optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    for i in range(cfg.epochs):
        train(model, optimizer, train_loader, device=device)
        acc = test(model, test_loader, device=device)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        if i % 5 == 0:
            # This saves the model to the trial directory
            with open(f"model_e{i}.pth", "wb") as f:
                torch.save(model.state_dict(), f)
