import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, metrics
import networkx as nx


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y)


def image_distance(x: torch.Tensor):
    """
    :param x: (B) x H x W x C
    :return: distances between pixels
    """
    if x.ndim == 3:
        x = x.flatten(0, 1)
    else:
        x = x.flatten(1, 2)

    return euclidean_dist(x, x)


def unique_points(x: np.array):
    return np.unique(x, axis=-2)


def compute_unique_distances(x: torch.Tensor):
    if x.ndim == 3:  # H x W x C
        x = x.flatten(0, 1)
    x = x.detach().numpy()
    x = torch.tensor(unique_points(x))  # N x C
    return euclidean_dist(x, x)


def draw_heatmap(d: torch.Tensor):
    fig, ax = plt.subplots()
    ax.imshow(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, d[i, j].item(), ha='center', va='center', color='w')
    fig.tight_layout()
    return fig


def show_results_ricci(G, name: str):
    # Print the first five results
    print(f"({name}) Graph, first 5 edges: ")
    for n1, n2 in list(G.edges())[:5]:
        print("Ollivier-Ricci curvature of edge (%s,%s) is %f" % (n1, n2, G[n1][n2]["ricciCurvature"]))

    fig, axes = plt.subplots(2, 1)

    # Plot the histogram of Ricci curvatures
    ax = axes[0]
    ricci_curvtures = nx.get_edge_attributes(G, "ricciCurvature").values()
    plt.hist(ricci_curvtures, bins=20)
    ax.set_xlabel('Ricci curvature')
    ax.set_title(f"Histogram of Ricci Curvatures ({name})")

    # Plot the histogram of edge weights
    ax = axes[1]
    weights = nx.get_edge_attributes(G, "weight").values()
    ax.hist(weights, bins=20)
    ax.set_xlabel('Edge weight')
    ax.set_title(f"Histogram of Edge weights ({name})")

    plt.tight_layout()
    return fig


def ARI(G, clustering, clustering_label="club"):
    """
    Computer the Adjust Rand Index (clustering accuracy) of "clustering" with "clustering_label" as ground truth.

    Parameters
    ----------
    G : NetworkX graph
        A given NetworkX graph with node attribute "clustering_label" as ground truth.
    clustering : dict or list or list of set
        Predicted community clustering.
    clustering_label : str
        Node attribute name for ground truth.

    Returns
    -------
    ari : float
        Adjust Rand Index for predicted community.
    """

    complex_list = nx.get_node_attributes(G, clustering_label)

    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))

    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1

    return metrics.adjusted_rand_score(y_true, y_pred)


def my_surgery(G_origin: nx.Graph(), weight="weight", cut=0):
    """A simple surgery function that remove the edges with weight above a threshold

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    weight: str
        The edge weight used as Ricci flow metric. (Default value = "weight")
    cut: float
        Manually assigned cutoff point.

    Returns
    -------
    G : NetworkX graph
        A graph after surgery.
    """
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Cut value should be greater than 0."
    if not cut:
        cut = (max(w.values()) - 1.0) * 0.6 + 1.0  # Guess a cut point as default

    to_cut = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cut:
            to_cut.append((n1, n2))
    print("*************** Surgery time ****************")
    print("* Cut %d edges." % len(to_cut))
    G.remove_edges_from(to_cut)
    print("* Number of nodes now: %d" % G.number_of_nodes())
    print("* Number of edges now: %d" % G.number_of_edges())
    cc = list(nx.connected_components(G))
    print("* Modularity now: %f " % nx.algorithms.community.quality.modularity(G, cc))
    print("* ARI now: %f " % ARI(G, cc))
    print("*********************************************")

    return G
