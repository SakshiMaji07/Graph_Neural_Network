from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Set the global random seed to improve reproducibility across the training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Research-grade training loop for the Siamese GNN graph-similarity model.

    The trainer orchestrates the full experiment lifecycle:
      1. Move graph pairs and targets to the selected device.
      2. Run a forward pass through the Siamese model.
      3. Compute the task loss and backpropagate it.
      4. Update optimizer parameters.
      5. Validate without gradient tracking to avoid leakage.
      6. Save the best checkpoint according to validation loss.
      7. Apply optional learning-rate scheduling and early stopping.

    The expected batch format is either a dictionary with keys ``"hypothesis"``,
    ``"ground_truth"``, and ``"target"`` or a 3-tuple of the form
    ``(hypothesis_graph, ground_truth_graph, target)``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable[..., torch.Tensor]] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device | str] = None,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        scheduler: Optional[Any] = None,
        mixed_precision: bool = False,
        save_dir: str | Path = "checkpoints",
        checkpoint_name: str = "best_model.pt",
        max_epochs: int = 100,
        patience: int = 5,
        min_delta: float = 1e-6,
        early_stopping: bool = True,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        if seed is not None:
            set_seed(seed)

        if model is None:
            raise ValueError("A PyTorch model must be supplied to the Trainer.")
        if train_loader is None:
            raise ValueError("A train dataloader must be supplied to the Trainer.")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.MSELoss()
        self.device = torch.device(device) if device is not None else self._auto_device()
        self.model.to(self.device)

        self.optimizer = optimizer or self._build_optimizer(
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = self._build_scheduler(scheduler)
        self.use_mixed_precision = bool(mixed_precision)
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision and self.device.type == "cuda")
            if self.device.type == "cuda"
            else None
        )

        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.early_stopping_enabled = bool(early_stopping)
        self.verbose = bool(verbose)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / checkpoint_name

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.epoch_metrics: list[dict[str, float | int]] = []
        self.training_history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        self.checkpoint_config: dict[str, Any] | None = None

        self._log(f"Trainer initialized on device: {self.device}")
        self._log(f"Optimizer: {type(self.optimizer).__name__}")
        self._log(f"Criterion: {type(self.criterion).__name__}")

    @staticmethod
    def _auto_device() -> torch.device:
        """Select CUDA if available, otherwise fall back to CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(
        self,
        optimizer_name: str,
        lr: float,
        weight_decay: float,
    ) -> Optimizer:
        """Construct a supported optimizer for the model parameters."""
        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported optimizers are: 'adam', 'adamw'."
        )

    def _build_scheduler(self, scheduler: Optional[Any]) -> Optional[Any]:
        """Accept either a scheduler instance or a configuration dict."""
        if scheduler is None:
            return None

        if isinstance(scheduler, dict):
            scheduler_name = str(scheduler.get("type") or scheduler.get("name") or "").lower()
            if scheduler_name == "step":
                return StepLR(
                    self.optimizer,
                    step_size=int(scheduler.get("step_size", 10)),
                    gamma=float(scheduler.get("gamma", 0.5)),
                )
            if scheduler_name in {"plateau", "reduce_on_plateau", "reduceonplateau"}:
                return ReduceLROnPlateau(
                    self.optimizer,
                    mode=str(scheduler.get("mode", "min")),
                    factor=float(scheduler.get("factor", 0.5)),
                    patience=int(scheduler.get("patience", 3)),
                    min_lr=float(scheduler.get("min_lr", 1e-6)),
                    threshold=float(scheduler.get("threshold", 1e-4)),
                )
            raise ValueError(
                "Unsupported scheduler configuration. Supported: 'step' and 'plateau'."
            )

        if not hasattr(scheduler, "step"):
            raise TypeError("The provided scheduler object does not implement a step method.")

        return scheduler

    def _log(self, message: str) -> None:
        """Emit human-readable training information."""
        if self.verbose:
            print(message)

    def _current_lr(self) -> float:
        """Return the current learning rate from the underlying optimizer."""
        for param_group in self.optimizer.param_groups:
            return float(param_group["lr"])
        return float("nan")

    def _autocast_context(self):
        """Create a mixed-precision context when enabled, otherwise a no-op."""
        if not self.use_mixed_precision:
            return nullcontext()

        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.device.type == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

        return nullcontext()

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accept either dict-based or tuple-based batch layouts and move them to the device."""
        if isinstance(batch, dict):
            if all(key in batch for key in ("hypothesis", "ground_truth", "target")):
                hypothesis = batch["hypothesis"]
                ground_truth = batch["ground_truth"]
                target = batch["target"]
            elif all(key in batch for key in ("graph_hypothesis", "graph_ground_truth", "similarity_target")):
                hypothesis = batch["graph_hypothesis"]
                ground_truth = batch["graph_ground_truth"]
                target = batch["similarity_target"]
            else:
                raise ValueError(
                    "Unsupported batch dictionary. Expected keys: "
                    "('hypothesis', 'ground_truth', 'target') or "
                    "('graph_hypothesis', 'graph_ground_truth', 'similarity_target')."
                )
        elif isinstance(batch, (tuple, list)) and len(batch) >= 3:
            hypothesis, ground_truth, target = batch[:3]
        else:
            raise TypeError(
                "Unsupported batch type. Expected a dict or a tuple/list of length at least 3."
            )

        hypothesis = hypothesis.to(self.device)
        ground_truth = ground_truth.to(self.device)
        target = torch.as_tensor(target, device=self.device, dtype=torch.float32)

        if target.ndim == 0:
            target = target.view(1)
        if target.ndim > 1:
            target = target.reshape(-1)

        return hypothesis, ground_truth, target

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Execute the user-defined criterion and normalize any tuple/dict outputs."""
        if predictions is None:
            raise ValueError("Model output cannot be None.")

        output = self.criterion(predictions, targets)

        if isinstance(output, dict):
            loss = output.get("loss")
            if loss is None:
                raise ValueError("Criterion dictionary output must contain a 'loss' key.")
            return loss

        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
            raise TypeError("Criterion returned a tuple/list without a tensor loss component.")

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Criterion must return a torch.Tensor, dict with a 'loss' key, or tuple/list containing a tensor."
            )

        return output

    def train_epoch(self) -> float:
        """Run one training epoch over the full training dataloader."""
        self.model.train()
        epoch_total_loss = 0.0
        num_examples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            hypothesis, ground_truth, targets = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_context():
                predictions = self.model(hypothesis, ground_truth)
                loss = self._compute_loss(predictions, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered during training at batch {batch_idx}. "
                    f"Loss value: {loss}"
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            batch_size = targets.shape[0]
            epoch_total_loss += loss.detach().item() * batch_size
            num_examples += batch_size

        if num_examples == 0:
            return 0.0

        return epoch_total_loss / num_examples

    def validate_epoch(self) -> float:
        """Run validation without gradients to avoid leaking validation information into training."""
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        epoch_total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                hypothesis, ground_truth, targets = self._prepare_batch(batch)

                with self._autocast_context():
                    predictions = self.model(hypothesis, ground_truth)
                    loss = self._compute_loss(predictions, targets)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss encountered during validation at batch {batch_idx}. "
                        f"Loss value: {loss}"
                    )

                batch_size = targets.shape[0]
                epoch_total_loss += loss.detach().item() * batch_size
                num_examples += batch_size

        if num_examples == 0:
            return 0.0

        return epoch_total_loss / num_examples

    def _maybe_step_scheduler(self, val_loss: float) -> None:
        """Update schedulers after each epoch. Plateau schedulers use validation loss."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False) -> None:
        """Persist a checkpoint describing the training state."""
        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "history": self.training_history,
            "seed": int(torch.random.initial_seed()),
            "config": self.checkpoint_config,
        }

        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            self._log(f"Best checkpoint saved to {self.checkpoint_path} at epoch {epoch}.")

    def fit(self, max_epochs: Optional[int] = None) -> dict[str, list[float]]:
        """Train the model for the requested number of epochs."""
        epochs_to_run = self.max_epochs if max_epochs is None else int(max_epochs)

        for epoch in range(self.current_epoch, epochs_to_run):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            self.current_epoch = epoch + 1
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["lr"].append(self._current_lr())

            epoch_metrics = {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self._current_lr(),
            }
            self.epoch_metrics.append(epoch_metrics)

            is_best = val_loss < self.best_val_loss - self.min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = self.current_epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch=self.current_epoch, train_loss=train_loss, val_loss=val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1

            self._maybe_step_scheduler(val_loss)

            self._log(
                "Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                "lr={lr:.6e} | best_val_loss={best_val_loss:.6f}".format(
                    epoch=self.current_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=self._current_lr(),
                    best_val_loss=self.best_val_loss,
                )
            )

            if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                self._log(
                    f"Early stopping triggered after {self.current_epoch} epochs with no improvement "
                    f"for {self.patience} validation epochs."
                )
                break

        return self.training_history


__all__ = ["Trainer", "set_seed"]
