from torch.nn import Module, Conv2d, Linear
from torch import relu, log_softmax, max_pool2d


class Model(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 3, kernel_size=3)
        self.fc = Linear(192, 10)

    def forward(self, x):
        x = relu(max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return log_softmax(x, dim=1)
