import torch
import torch.nn as nn
import torch.nn.functional as F
from compression import batch_compression_curvature, batch_complexity


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
        ], dim=0).to('cuda')

        if x.shape[1] < self.emb.weight.shape[1]:
            x = F.pad(x, (0, 0, 0, self.emb.weight.shape[1] - x.shape[1]), value=len(self.abc))

        x = self.conv2(self.conv1(self.emb(x).transpose(-1, -2))).transpose(-1, -2)
        _, d, v = torch.linalg.svd(x, full_matrices=False)
        x = torch.bmm(d.unsqueeze(-1).expand(-1, -1, v.shape[1]), v)

        x = self.linear(self.flatten(self.conv22d(self.conv12d(x.unsqueeze(-3)))))
        return x if logits else self.softmax(x)

    def training_step(self, X: list[str], y: torch.Tensor):
        weight = self.w(X).to('cuda')
        o = self(X)
        loss = F.cross_entropy(o, y, reduction='none') * weight
        return loss.sum()
