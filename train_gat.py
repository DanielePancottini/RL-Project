import torch

import torch.nn.functional as F

from model import GNNWithAttention

class TrainGAT:
    def __init__(self, dataset, graph_data):
        self.data = graph_data    
        self.in_channels = self.data.num_node_features
        self.hidden_channels = 128
        self.out_channels = 64 
        self.num_heads = 8
        self.num_classes = dataset.num_classes 
        self.num_epochs = 100
        self.weight_decay = 1e-2
        self.lr = 1e-2

        self.model = GNNWithAttention(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
            num_heads = self.num_heads,
            num_classes = self.num_classes
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5, verbose=True)

    def train_gat(self):

        self.model.train()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            output = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(output[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            # Validation loss
            val_loss = self._evaluate_loss(self.model, self.data, mask=self.data.val_mask)

            # Scheduler step
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

        return self.model

    def evaluate_accuracy(self, model, data):
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            preds = output[data.test_mask].max(1)[1]
            correct = preds.eq(data.y[data.test_mask]).sum().item()
            accuracy = correct / data.test_mask.sum().item()
        return accuracy

    def _evaluate_loss(self, model, data, mask):
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            loss = F.cross_entropy(output[mask], data.y[mask])
        return loss.item()

   