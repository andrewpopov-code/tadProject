import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd


class LinearRegression(nn.Module):
    def __init__(self, m: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(m, 1, bias=bias)

    def forward(self, X):
        return self.linear(X)

    def loss(self, y_pred, y):
        return torch.square(y - y_pred).sum()


class WeightedLinearRegression(nn.Module):
    def __init__(self, m: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(m, 1, bias=bias)

    def forward(self, X):
        return self.linear(X)

    def loss(self, y_pred, y, w: torch.Tensor):
        return torch.sum(torch.square(y - y_pred) * w)


def collate_fn(items: list[np.ndarray[float]]):
    items = np.vstack(items)
    y = torch.tensor(items[:, ix]).to(device)
    X = torch.tensor(np.delete(items, ix, axis=1)).to(device)
    return X, y


def collate_fn_weighted(items: list[np.ndarray[float]]):
    items = np.vstack(items)
    y = torch.tensor(items[:, ix]).to(device)
    X = torch.tensor(np.delete(items, ix, axis=1)).to(device)
    w = 1 / (torch.tensor(np.square(y - y.mean())) + 1e-7)
    return X, y, w.to(device)


def collate_fn_weighted2(items: list[np.ndarray[float]]):
    items = np.vstack(items)
    y = torch.tensor(items[:, ix]).to(device)
    X = torch.tensor(np.delete(items, ix, axis=1)).to(device)
    w = 1 / (torch.tensor(y / (y.min() + 1e-7)) + 1e-7)
    return X, y, w.to(device)


device = torch.device('cuda')
data: pd.DataFrame
ix: int

# Dataset X_train + y_train merged
dataloader = DataLoader(
    dataset=data.to_numpy(),
    batch_size=64,
    collate_fn=collate_fn
)

learning_rate = 1e-3
batch_size = 64
epochs = 5
model = LinearRegression(m=10, bias=False)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train_loop():
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = model.loss(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop()
print("Done!")
