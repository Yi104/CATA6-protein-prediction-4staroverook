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
        ic_path: str = "checkpoints/go_ic.pt",
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.model.to(self.device)
        self.idx2go = idx2go
        self.godag = godag
        self.go_ic = None
        self.ic_meta = None

        # -------------------------------
        # Load GO IC weights (metric only)
        # -------------------------------
        if ic_path and os.path.exists(ic_path):
            payload = torch.load(ic_path, map_location="cpu")

            # Backward compatible:
            # - old format: payload is a dict {go_id: ic}
            # - new format: payload is {"go_ic": {...}, ...meta...}
            if isinstance(payload, dict) and "go_ic" in payload:
                self.go_ic = payload["go_ic"]
                self.ic_meta = {k: v for k, v in payload.items() if k != "go_ic"}
            else:
                self.go_ic = payload
                self.ic_meta = {"format": "legacy_dict"}

            print(f"Loaded GO IC weights from {ic_path} (n_terms={len(self.go_ic)})")
        else:
            print("GO IC not found (or disabled), using unweighted metric")

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
                              top_k=200,
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

        metrics = compute_fmax_hierarchical(
            logits,
            targets,
            idx2go,
            self.godag,
            go_ic=self.go_ic,  # key points
            thresholds=thresholds,
            max_proteins=max_proteins,
            top_k=top_k,
            return_best_threshold = True,
        )
        # Backward/forward compatible unpacking:
        if isinstance(metrics, dict):
            # Prefer unweighted for checkpoint selection
            fmax = float(metrics.get("fmax_unweighted", 0.0))
            best_t = float(metrics.get("best_t_unweighted", 0.0))
        else:
            # Old API: (fmax, best_t)
            fmax, best_t = metrics


        avg_loss = total_loss / len(dataloader)
        print("metric mode:", "IC-weighted" if self.go_ic is not None else "unweighted")
        return avg_loss, fmax, best_t
