import torch

import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

class Trainer:
    def __init__(self, model, class_weights, device):

        self.device = device

        # Training params
        self.num_epochs = 10
        self.weight_decay = 1e-4
        self.lr = 1e-5
        
        # Model & Optimizer
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    def train(self, train_loader, val_loader):
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            total_samples = 0

            for batch_idx, data in enumerate(train_loader):
                
                data = data.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data.x, data.edge_index, data.batch)
                target = data.y.squeeze().long()

                # Calculate loss
                loss = self.loss_fn(output, target) 

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()

                # Accumulate loss
                total_loss += loss.item() * data.y.size(0)
                total_samples += data.y.size(0)

            # Average training loss
            avg_train_loss = total_loss / total_samples if total_samples > 0 else float('inf')

            # Validation loss
            val_loss = self._evaluate_loss(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self.model

    def evaluate_metrics(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Probabilities for AUC-ROC

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data.x, data.edge_index, data.batch)
                
                # Convert logits to probabilities for AUC-ROC
                probs = torch.sigmoid(output[:, 1])  # Get probability of class 1

                # Convert probabilities to binary class predictions
                preds = (probs > 0.5).long()  # Threshold at 0.5
                
                all_probs.extend(probs.cpu().numpy())  # Probabilities for AUC-ROC
                all_preds.extend(preds.cpu().numpy())  # Class predictions
                all_labels.extend(data.y.cpu().numpy())  # True labels

        # Calculate metrics
        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Compute AUC-ROC
        auc_roc = roc_auc_score(all_labels, all_probs)

        return accuracy, precision, recall, f1, conf_matrix, auc_roc


    def _evaluate_loss(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for data in test_loader:

                data = data.to(self.device)

                output = self.model(data.x, data.edge_index, data.batch)
                target = data.y.squeeze().long()

                loss = self.loss_fn(output, target)

                total_loss += loss.item()
                total_samples += data.y.size(0)

        return total_loss / total_samples if total_samples > 0 else float('inf')

   