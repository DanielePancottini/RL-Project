import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from env import GNNInterpretEnvironment
from datasets.ground_truth_loader import load_dataset_ground_truth

def evaluate_policy_ba2(model, policy, test_dataset, test_indices, device):
    
    print("Loading ground-truth explanations for test indices...")
    (gt_edge_indices, gt_edge_labels), _ = load_dataset_ground_truth(
        "ba2", shuffle=False, test_indices=test_indices
    )

    gt_edge_indices = [gt_edge_indices[i] for i in test_indices]
    gt_edge_labels = [gt_edge_labels[i] for i in test_indices]

    # Create an environment for rollout
    env = GNNInterpretEnvironment(
        gnn_model=model, 
        dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False),
        max_steps=20, 
        device=device
    )

    # Alignment check for test_dataset and ground truth
    assert len(test_dataset) == len(gt_edge_indices) == len(gt_edge_labels), f"Length mismatch: test_dataset={len(test_dataset)}, gt_edge_indices={len(gt_edge_indices)}, gt_edge_labels={len(gt_edge_labels)}"
    for i in range(len(test_dataset)):
        graph = test_dataset[i]
        edge_index = gt_edge_indices[i]
        # Example: compare number of edges
        if graph.edge_index.shape[1] != edge_index.shape[1]:
            print(f"[Alignment Warning] Graph {i}: test_dataset edges={graph.edge_index.shape[1]}, ground truth edges={edge_index.shape[1]}")
        # Optionally, compare number of nodes
        test_nodes = set(graph.edge_index.flatten().cpu().numpy())
        gt_nodes = set(edge_index.flatten())
        if test_nodes != gt_nodes:
            print(f"[Alignment Warning] Graph {i}: node sets differ between test_dataset and ground truth.")

    auc_scores = []

    print("Evaluating policy deterministically (stochastic=False)...")
    for i in tqdm(range(len(test_dataset)), desc="Evaluating on test graphs"):
        
        # Properly reset environment to initialize everything
        obs, _ = env.reset()   # sets current_graph, original_pred_logits, etc.

        # perform deterministic rollout
        reward, info, S_pred = env.simulate_rollout_from_S(
            S_init=env.S, policy=policy, stochastic=False
        )

        print("GT edge shape:", gt_edge_indices[i].shape, len(gt_edge_labels[i]))
        print("GT nodes:", len(np.unique(gt_edge_indices[i])))
        print("Test Dataset Graph Info:", env.current_graph)
        print("Predicted nodes:", S_pred)

        # load corresponding ground-truth edges & labels
        edge_index = gt_edge_indices[i]  # shape (2, num_edges)
        gt_labels = gt_edge_labels[i]    # 0/1 ground truth edge membership

        # predict edges based on whether both endpoints are in S_pred
        preds = np.array([1 if (u in S_pred and v in S_pred) else 0 for u, v in edge_index.T])

        # compute AUC
        try:
            auc = roc_auc_score(gt_labels, preds)
            auc_scores.append(auc)
        except ValueError:
            # happens if all gt_labels are 0 or 1
            continue

    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    print(f"\n✅ Average Explanation AUC across test graphs: {mean_auc:.4f}")
    return mean_auc