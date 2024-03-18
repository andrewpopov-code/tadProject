import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y)


def draw_heatmap(d: torch.Tensor, writer: SummaryWriter, tag: str, title: str):
    fig, ax = plt.subplots()
    ax.imshow(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, d[i, j], ha='center', va='center', color='w')
    fig.tight_layout()

    writer.add_figure('/'.join((tag, title)), fig)
