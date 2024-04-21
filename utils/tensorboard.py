import torch
import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams


def draw_heatmap(d: np.ndarray):
    """
    :param d: a matrix
    :return: a figure with a heatmap
    """
    fig, ax = plt.subplots()
    ax.imshow(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, d[i, j], ha='center', va='center', color='w')
    fig.tight_layout()
    return fig


def plot_persistence(dgm: list):
    """
    :param dgm: persistence diagrams of kind: dimension x birth time x death time
    :return: figure with diagrams drawn
    """
    # TODO: make our own version
    fig = plt.figure()
    plot_diagrams(dgm)
    return fig


def plot_persistence_each(dgms: list) -> plt.Figure:
    """
    :param dgms: sets of persistence diagrams of kind: dimension x birth time x death time
    :return: figure with all diagrams plotted separately
    """
    fig, _ = plt.subplots(1, len(dgms))
    for i in range(len(dgms)):
        plt.subplot(1, len(dgms), i + 1)
        plot_diagrams(dgms[i])
    return fig


def plot_betti(bc: np.ndarray, ax: plt.Axes = None) -> [plt.Figure, plt.Axes]:
    """
    :param bc: a set of betti curves
    :param ax: canvas to use
    :return: a new figure or the ax with betti curves drawn
    """
    ax = ax or plt.figure()
    for dim in range(bc.shape[0]):
        plt.plot(bc[dim])
    return ax


def plot_betti_each(bc: list[np.ndarray]) -> plt.Figure:
    """
    :param bc: list of sets of betti curves
    :return: figure with all sets plotted separately
    """
    fig, axes = plt.subplots(1, len(bc))
    for i in range(len(bc)):
        plot_betti(bc[i], axes[i])
    return fig
