import pandas as pd
import seaborn as sns
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


def plot_persistence(dgm: list[np.ndarray]):
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


def plot_persistence_pdf(dgm: list[np.ndarray]) -> plt.Figure:
    """
    :param dgm: persistence diagrams of kind: dimension x birth time x death time
    :return: figure with plotted cdf's
    """
    l: list[np.ndarray] = [dgm[i][1] - dgm[i][0] for i in range(len(dgm))]
    fig, axes = plt.subplots(1, len(dgm))
    for i in range(len(dgm)):
        axes[i].plot(np.cumsum(l[i].sort()) / np.sum(l[i]))
        axes[i].set_title(f'H{i}')
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


def plot_dimension(dim: list[np.ndarray], columns: list[str]) -> plt.Figure:
    """
    :param dim: list of different dimension estimates
    :param columns: labels of estimates
    :return: figure with the pairwise comparison chart
    """
    return sns.pairplot(pd.DataFrame(dim, columns=columns).T, diag_kind='hist').figure


def plot_andrews(X: np.ndarray, ax: plt.Axes = None, c: str = 'b', pts: int = 100) -> plt.Figure:
    d = X.shape[1]
    v = np.zeros((2 * d + 1, pts))
    v[0, :] = 1 / np.sqrt(2)
    t = np.linspace(-np.pi, np.pi, endpoint=True, num=pts)
    v[1::2, :] = np.sin(np.arange(d).reshape(-1, 1) * t)
    v[2::2, :] = np.cos(np.arange(d).reshape(-1, 1) * t)
    y = X @ v[:d]  # d x pts, X - n x d

    ax = ax or plt.figure()
    ax.plot(*[x for p in zip([t] * X.shape[0], y) for x in p], c=c)
    return ax
