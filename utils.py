from torch_geometric.data import Data

def get_max_nodes_edges_from_dataset(dataset):
    """
    Computes the max number of nodes and edges in any *single graph*
    within the provided dataset, and the node feature dimension.
    """
    max_nodes = 0
    max_edges = 0
    node_feature_dim = 0

    print("Calculating true max nodes/edges from the raw dataset (not DataLoader batches)...")
    for i, data in enumerate(dataset):
        if isinstance(data, Data):
            max_nodes = max(max_nodes, data.num_nodes)
            max_edges = max(max_edges, data.num_edges)
            if node_feature_dim == 0 and data.x is not None:
                node_feature_dim = data.x.shape[1] # Assuming x is always present and has features
        else:
            print(f"Warning: Dataset element {i} is not a PyG Data object. Type: {type(data)}. Skipping.")
    
    if node_feature_dim == 0:
        raise ValueError("Could not determine node feature dimension. Ensure dataset contains PyG Data objects with node features (data.x).")

    print(f"Dataset-wide Max Nodes: {max_nodes}, Max Edges: {max_edges}, Node Features: {node_feature_dim}")
    return max_nodes, max_edges, node_feature_dim