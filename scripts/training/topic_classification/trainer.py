import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CustomTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        loss_fn,
        optimizer,
        batch_size=8,
        device=None,
    ):
        """
        Custom Trainer for training and evaluating a model.

        Args:
            model: The model to train.
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
            loss_fn: Loss function for training.
            optimizer: Optimizer for training.
            batch_size: Batch size for training and evaluation.
            device: Device to use ('cpu' or 'cuda').
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.history = {"train_loss": [], "eval_loss": []}
        self.model.to(self.device)

    def create_batches(self, dataset):
        """
        Manually creates batches of data from the dataset.

        Args:
            dataset: Dataset to create batches from.

        Returns:
            Generator that yields batches of data.
        """
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i : i + self.batch_size]
            inputs = {
                key: torch.tensor(batch[key]).to(self.device) for key in batch
            }
            yield inputs

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            The average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0

        epoch_iterator = tqdm(
            self.create_batches(self.train_dataset),
            desc=f"Epoch {epoch}",
            ncols=100,
        )

        for batch in epoch_iterator:
            self.optimizer.zero_grad()
            labels = batch.pop("labels")

            logits = self.model(**batch)

            loss = self.loss_fn(logits, labels)
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            epoch_iterator.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(self.train_dataset)
        return avg_loss

    def evaluate(self):
        """
        Evaluate the model on the validation dataset.

        Returns:
            The average evaluation loss.
        """
        self.model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for batch in self.create_batches(self.eval_dataset):
                labels = batch.pop("labels")

                logits = self.model(**batch)

                loss = self.loss_fn(logits, labels)
                eval_loss += loss.item()

        avg_loss = eval_loss / len(self.eval_dataset)
        return avg_loss

    def train(self, epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs: Number of epochs to train the model.
        """
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs}")

            train_loss = self.train_one_epoch(epoch + 1)

            eval_loss = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["eval_loss"].append(eval_loss)

            print(f"Epoch {epoch + 1}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss: {eval_loss:.4f}")

        print("Training complete!")

    def plot_training_history(self):
        """
        Plot the training history.
        """
        plt.plot(self.history["train_loss"], label="train loss")
        plt.plot(self.history["eval_loss"], label="eval loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./model/output/loss_fig.png")
