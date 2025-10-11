import os
import random

from numpy.random.mtrand import RandomState
import torch

from datasets.utils import preprocess_adj, adj_to_edge_index, load_real_dataset, get_graph_data

import pickle as pkl
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
import collections


def load_ba2_ground_truth(shuffle=True):
    """Load a the ground truth from the ba2motif dataset.

    :param shuffle: Wheter the data should be shuffled.
    :returns: np.array, np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
    with open(path, 'rb') as fin:
        adjs, features, labels = pkl.load(fin)

    n_graphs = adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42)  # Make sure that the permutation is always the same, even if we set the seed different
        shuffled_indices = prng.permutation(indices)
    else:
        shuffled_indices = indices

    # Create shuffled data
    shuffled_adjs = adjs[shuffled_indices]
    shuffled_edge_index = adj_to_edge_index(shuffled_adjs)

    np_edge_labels = []

    # Obtain the edge labels.
    insert = 20
    skip = 5
    for edge_index in shuffled_edge_index:
        labels = []
        for pair in edge_index.T:
            r = pair[0]
            c = pair[1]
            # In line with the original PGExplainer code we determine the ground truth based on the location in the index
            if r >= insert and r < insert + skip and c >= insert and c < insert + skip:
                labels.append(1)
            else:
                labels.append(0)
        np_edge_labels.append(np.array(labels))

    return shuffled_edge_index, np_edge_labels


def load_mutag_ground_truth(shuffle=True):
    """Load a the ground truth from the mutagenicity dataset.
    Mutag is a large dataset and can thus take a while to load into memory.
    
    :param shuffle: Wheter the data should be shuffled.
    :returns: np.array, np.array, np.array, np.array
    """
    print("Loading MUTAG dataset, this can take a while")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + "Mutagenicity" + '.pkl'
    if not os.path.exists(path): # pkl not yet created
        original_adjs, original_features, original_labels = load_real_dataset(path, dir_path + '/Mutagenicity/Mutagenicity_')
    else:
        with open(path, 'rb') as fin:
            original_adjs, original_features, original_labels = pkl.load(fin)

    print("Loading MUTAG groundtruth, this can take a while")
    path = dir_path + '/Mutagenicity/Mutagenicity_'
    edge_lists, _, edge_label_lists, _ = get_graph_data(path)

    n_graphs = original_adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
        shuffled_indices = prng.permutation(indices)
    else:
        shuffled_indices = indices

    # Create shuffled data
    shuffled_adjs = original_adjs[shuffled_indices]
    shuffled_labels = original_labels[shuffled_indices]
    shuffled_edge_list = [edge_lists[i] for i in shuffled_indices]
    shuffled_edge_label_lists = [edge_label_lists[i] for i in shuffled_indices]

    # Transform to edge index
    shuffled_edge_index = adj_to_edge_index(shuffled_adjs)

    return shuffled_edge_index, shuffled_labels, shuffled_edge_list, shuffled_edge_label_lists


def _load_node_dataset_ground_truth(_dataset):
    """Load a the ground truth from a synthetic node dataset.
    Mutag is a large dataset and can thus take a while to load into memory.
    
    :param shuffle: Whether the data should be shuffled.
    :returns: np.array, np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, _, _, _, _, _, _, _, edge_label_matrix  = pkl.load(fin)
    graph = preprocess_adj(adj)[0].astype('int64').T
    labels = []
    for pair in graph.T:
        labels.append(edge_label_matrix[pair[0], pair[1]])
    labels = np.array(labels)
    return graph, labels


def load_dataset_ground_truth(_dataset, shuffle=True, test_indices=None):
    """Load a the ground truth from a dataset.
    Optionally we can only request the indices needed for testing.
    
    :param test_indices: Only return the indices used by the PGExplaier paper.
    :returns: (np.array, np.array), np.array
    """
    if _dataset == "syn1" or _dataset == "syn2":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        if test_indices is None:
            return (graph, labels), range(400, 700, 5)
        else:
            all = range(400, 700, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    if _dataset == "syn3":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        if test_indices is None:
            return (graph, labels), range(511,871,6)
        else:
            all = range(511, 871, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    if _dataset == "syn4":
        graph, labels = _load_node_dataset_ground_truth(_dataset)
        if test_indices is None:
            return (graph, labels), range(511,800,1)
        else:
            all = range(511, 800, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    elif _dataset == "ba2":
        edge_index, labels = load_ba2_ground_truth(shuffle)
        allnodes = [i for i in range(0,100)]
        allnodes.extend([i for i in range(500,600)])
        if test_indices is None:
            return (edge_index, labels), allnodes
        else:
            all = range(0, 1000, 1)
            filtered = [i for i in all if i in test_indices]
            return (edge_index, labels), filtered
    elif _dataset == "mutag":
        edge_index, labels, edge_list, edge_labels = load_mutag_ground_truth(shuffle)
        selected = []
        np_edge_list = []
        for gid in range(0, len(edge_index)):
            ed = edge_list[gid]
            ed_np = np.array(ed).T
            np_edge_list.append(ed_np)
            if np.argmax(labels[gid]) == 0 and np.sum(edge_labels[gid]) ==4:
                selected.append(gid)
        np_edge_labels = [np.array(ed_lab) for ed_lab in edge_labels]
        if test_indices is None:
            return (np_edge_list, np_edge_labels), selected
        else:
            all = range(400, 700, 1)
            filtered = [i for i in all if i in test_indices]
            return (np_edge_list, np_edge_labels), filtered
    else:
        print("Dataset does not exist")
        raise ValueError
    
def generate_pretraining_samples(dataset, train_indices):
    """Generate pretraining samples (graph, seed_node, BFS_sequence) for each graph.

    For each graph in `train_indices` we optionally subsample seed nodes and
    extract their k-hop neighborhood (default 3). The neighborhood nodes are
    ordered by BFS starting from the seed node to form the target construction
    sequence tau'. Returned list contains tuples (graph.cpu(), seed_node, bfs_seq).
    """

    pretraining_samples = []

    # Configuration
    hops = 3
    max_seeds_per_graph = 1  # set to int to subsample seeds (e.g., 20)

    # Iterate graphs referenced by train_indices
    for idx in tqdm(train_indices, desc="Generating Pretraining Samples"):
        g_data = dataset[idx]
        # Convert to networkx for neighborhood and BFS operations
        g_nx = to_networkx(g_data, to_undirected=True)

        # All nodes in the graph
        all_nodes = list(g_nx.nodes())
        if not all_nodes:
            continue

        # Optionally subsample seed nodes for speed
        seed_nodes = all_nodes
        if isinstance(max_seeds_per_graph, int) and max_seeds_per_graph > 0:
            seed_nodes = random.sample(all_nodes, min(max_seeds_per_graph, len(all_nodes)))

        for seed in seed_nodes:
            # Get k-hop neighborhood (including seed)
            # Use BFS tree levels to compute nodes within `hops` distance
            bfs_tree = nx.single_source_shortest_path_length(g_nx, seed, cutoff=hops)
            neighborhood_nodes = set(bfs_tree.keys())
            if len(neighborhood_nodes) <= 1:
                continue

            # Create subgraph induced by the neighborhood
            sub_nx = g_nx.subgraph(neighborhood_nodes).copy()

            # Generate BFS ordering starting from seed to produce the sequence tau'
            bfs_order = list(nx.bfs_tree(sub_nx, source=seed))

            # Convert to list of ints (NetworkX nodes may be ints already)
            bfs_sequence = [int(n) for n in bfs_order]

            # Store the sample: keep the original PyG graph, seed and sequence
            pretraining_samples.append((g_data.cpu(), int(seed), bfs_sequence))

    print(f"Generated {len(pretraining_samples)} pretraining samples.")
    return pretraining_samples