from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.sum()
        return x
