from gtda.images import ImageToPointCloud, RadialFiltration
from utils import euclidean_dist
import torch


def image_distance(x: torch.Tensor):
    """
    :param x: B x H x W x C
    :return:
    """
    x = x.flatten(0, 2)
    return euclidean_dist(x, x)
