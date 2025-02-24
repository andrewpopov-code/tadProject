import torch
import torch.nn as nn
import torch.nn.functional as F
from src.intrinsic.functional.compression import batch_compression_curvature, batch_complexity


# class GeneClassifier2(nn.Module):
#     def __init__(self, nucleotides: list, n_classes: int, hidden_dim1: int = 128, hidden_dim2: int = 64, k: int = 5, weight_strategy: str = 'curvature'):
#         super().__init__()
#         self.abc, self.k = nucleotides, k
#         self.fc1 = nn.LazyLinear(hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, n_classes)
#         self.softmax = nn.Softmax(-1)
#         self.w = self.curvature if weight_strategy == 'curvature' else lambda x: torch.ones(len(x))
#
#     def curvature(self, X: list[str]):
#         weight = torch.tensor(batch_compression_curvature(X, self.abc, self.k))
#         return F.softmax(-weight / (weight.max() - weight.min()), dim=-1)
#
#     def complexity(self, X: list[str]):
#         weight = torch.tensor(batch_complexity(X, self.abc))
#         return F.softmax(-weight / (weight.max() - weight.min()), dim=-1)
#
#     def forward(self, x: torch.Tensor):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.softmax(self.fc3(x))
#
#     def training_step(self, X: list[str], bowX: torch.Tensor, y: torch.Tensor):
#         weight = self.w(X)
#         o = self(bowX)
#         loss = F.cross_entropy(o, y, reduction='none') * weight
#         return loss.sum()
#
#
# class GeneClassifier1(nn.Module):
#     def __init__(self, nucleotides: list, n_classes: int, emb_dim: int = 32, lstm_dim: int = 64, k: int = 5):
#         super().__init__()
#         self.abc, self.k = nucleotides, k
#         self.emb = nn.Embedding(len(nucleotides) + 1, emb_dim)
#         self.lstm = nn.LSTM(emb_dim, lstm_dim, batch_first=True)
#         self.proj = nn.Linear(lstm_dim, n_classes)
#         self.softmax = nn.Softmax(-1)
#
#     def forward(self, X: list[str], logits: bool = True):
#         l = len(max(X, key=len))
#         X = torch.stack([F.pad(torch.LongTensor([self.abc.index(x) for x in s]), pad=(0, l - len(s)), value=len(self.abc)) for s in X], dim=0)
#         X = self.proj(self.lstm(self.emb(X))[1][1][0])
#         return X if logits else self.softmax(X)
#
#     def training_step(self, X: list[str], y: torch.Tensor):
#         weight = torch.tensor(batch_compression_curvature(X, self.abc, self.k))
#         weight = F.softmax(-weight / (weight.max() - weight.min()), dim=-1)
#         X = self(X)
#         loss = F.cross_entropy(X, y, reduction='none') * weight / weight.sum()
#         return loss.sum()


class GeneClassifier(nn.Module):
    def __init__(self, nucleotides: list, n_classes: int, emb_dim: int = 32, k: int = 5, weight_strategy: str = 'curvature'):
        super().__init__()

        self.abc, self.k = nucleotides, k
        if weight_strategy == 'curvature':
            self.w = self.curvature
        elif weight_strategy == 'complexity':
            self.w = self.complexity
        else:
            self.w = lambda x: torch.ones(len(x))

        self.emb = nn.Embedding(len(nucleotides) + 1, emb_dim)
        self.conv1 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, emb_dim // 2, padding='same'),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, emb_dim // 2, padding='same'),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.conv12d = nn.Sequential(
            nn.Conv2d(1, emb_dim, emb_dim // 2),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        self.conv22d = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim * 2, emb_dim // 4),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        self.linear = nn.LazyLinear(n_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(-1)

    def curvature(self, X: list[str]):
        weight = torch.tensor(batch_compression_curvature(X, self.abc, self.k))
        return F.softmax(weight / (weight.max() - weight.min()), dim=-1)

    def complexity(self, X: list[str]):
        weight = torch.tensor(batch_complexity(X, self.abc))
        return F.softmax(-weight / (weight.max() - weight.min()), dim=-1)

    def forward(self, x: list[str], logits: bool = True):
        l = len(max(x, key=len))
        x = torch.stack([
            F.pad(torch.LongTensor([self.abc.index(x) for x in s]), pad=(0, l - len(s)), value=len(self.abc)) for s in x
        ], dim=0)

        if x.shape[1] < self.emb.weight.shape[1]:
            x = F.pad(x, (0, 0, 0, self.emb.weight.shape[1] - x.shape[1]), value=len(self.abc))

        x = self.conv2(self.conv1(self.emb(x).transpose(-1, -2))).transpose(-1, -2)
        _, d, v = torch.linalg.svd(x, full_matrices=False)
        x = torch.bmm(d.unsqueeze(-1).expand(-1, -1, v.shape[1]), v)

        x = self.linear(self.flatten(self.conv22d(self.conv12d(x.unsqueeze(-3)))))
        return x if logits else self.softmax(x)

    def training_step(self, X: list[str], y: torch.Tensor):
        weight = self.w(X)
        o = self(X)
        loss = F.cross_entropy(o, y, reduction='none') * weight
        return loss.sum()
