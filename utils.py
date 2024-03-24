import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y)


def image_distance(x: torch.Tensor):
    """
    :param x: H x W x C
    :return:
    """
    x = x.flatten(0, 1)
    return euclidean_dist(x, x)


def draw_heatmap(d: torch.Tensor, writer: SummaryWriter, tag: str, title: str):
    fig, ax = plt.subplots()
    ax.imshow(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, d[i, j], ha='center', va='center', color='w')
    fig.tight_layout()

    writer.add_figure('/'.join((tag, title)), fig)


def betti_curves_str(dim: int, outer_step: [int, str], inner_step: [int, str]):
    return f'Betti Curves/Step {outer_step}/Layer {inner_step} Betti Curve/Dimension {dim}'


def id_str(outer_step: [int, str], inner_step: [int, str]):
    return f'ID Profile/Step {outer_step}/Layer {inner_step}'
