import torch
from tqdm import tqdm
from src.evaluation.metrics import compute_fmax

class Trainer:
    """
    Generic trainer for multi-label classification.

    Responsibilities:
    - one training epoch
    - one validation epoch
    - loss computation
    - device handling

    Does NOT:
    - load data
    - define model
    - save checkpoints
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.model.to(self.device)

    def train_one_epoch(self, dataloader):
        """
        Train for a single epoch.

        Returns
        -------
        avg_loss : float
        """
        self.model.train()
        total_loss = 0.0

        for x, y in tqdm(dataloader, desc="Training", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate_one_epoch(self, dataloader):
        """
        Run validation (no gradient updates).

        Returns
        -------
        avg_loss : float
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Validation", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate_with_metrics(self, dataloader):
        """
        Validation pass that computes loss and Fmax.

        Returns
        -------
        avg_loss : float
        fmax : float
        best_threshold : float
        """

        self.model.eval()
        total_loss = 0.0

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_targets.append(y.cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        fmax, best_t = compute_fmax(logits, targets)

        avg_loss = total_loss / len(dataloader)
        return avg_loss, fmax, best_t
