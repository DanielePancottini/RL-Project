def _choose_seed_gradient(self, graph):
    graph.x.requires_grad = True
    self.gnn_model.eval()

    logits = self.gnn_model(graph.x, graph.edge_index).squeeze(0)
    target_class = logits.argmax()
    
    self.gnn_model.zero_grad()
    logits[target_class].backward()
    
    # Node importance = L2 norm of gradients
    node_importance = graph.x.grad.norm(dim=1)
    seed_node = int(torch.argmax(node_importance).item())
    
    graph.x.requires_grad = False
    return seed_node