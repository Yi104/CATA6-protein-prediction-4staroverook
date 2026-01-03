## src/training/trainer.py

import os
import torch
from tqdm import tqdm
from src.evaluation.metrics import compute_fmax_hierarchical

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
        idx2go: list,
        godag,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.model.to(self.device)
        self.idx2go = idx2go
        self.godag = godag

        # -------------------------------
        # Load GO IC weights (metric only)
        # -------------------------------
        ic_path = "checkpoints/go_ic.pt"
        if os.path.exists(ic_path):
            self.go_ic = torch.load(ic_path)
            print("Loaded GO IC weights")
        else:
            self.go_ic = None
            print("GO IC not found, using unweighted metric")

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

    def validate_with_metrics(self, dataloader,
                              thresholds=None, max_proteins=1000,
                              topk=200,
                              term_idx= None  #for ontology subset evaluation
    ):
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
            #for x, y in tqdm(dataloader, desc="Val+Metrics", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_targets.append(y.cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # slice ontology subset if provided:
        if term_idx is not None:
            logits = logits[:,term_idx]
            targets = targets[:,term_idx]
            idx2go = [self.idx2go[i] for i in term_idx]
        else:
            idx2go = self.idx2go

        fmax, best_t = compute_fmax_hierarchical(
            logits,
            targets,
            idx2go,
            self.godag,
            go_ic=self.go_ic,  # key points
            thresholds=thresholds,
            max_proteins=max_proteins,
            topk=topk,
        )

        avg_loss = total_loss / len(dataloader)
        return avg_loss, fmax, best_t
