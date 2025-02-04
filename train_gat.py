import torch

import torch.nn.functional as F

class Trainer:
    def __init__(self, model, device):

        self.device = device

        # Training params
        self.num_epochs = 10
        self.batch_size = 32
        self.weight_decay = 1e-2
        self.lr = 1e-2
        
        # Model & Optimizer
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    def train(self, train_loader, test_loader):
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0

            for data in train_loader:
                
                data = data.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, data.y)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Validation loss
            val_loss = self._evaluate_loss(test_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self.model

    def evaluate_accuracy(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                preds = output.argmax(dim=1)  # Get class with max probability
                correct += (preds == data.y).sum().item()
                total += data.y.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def _evaluate_loss(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = F.cross_entropy(output, data.y)
                total_loss += loss.item()

        return total_loss / len(test_loader)

   