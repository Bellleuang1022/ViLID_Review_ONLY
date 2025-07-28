import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f"training_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###################################
# EARLY STOPPING CLASS
###################################
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 5.
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model, rank=None):
        """
        Checks whether validation loss has improved enough; if not, increase the counter.
        If the counter exceeds self.patience, trigger early stop.
        The optional 'rank' parameter is ignored.
        """
        if val_loss < self.best_score - self.delta:
            if self.verbose:
                logger.info(f"EarlyStopping: Validation loss improved from {self.best_score:.4f} to {val_loss:.4f}")
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


###################################
# Checkpoint function
###################################
def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> None:
    """
    Save a training checkpoint.

    Args:
        epoch (int): Current epoch.
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        loss (float): Validation (or best) loss.
        path (str): File path to save the checkpoint.
        scheduler (optional): Learning-rate scheduler state.
        scaler (optional): GradScaler state for mixed precision.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f} to {path}")

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cpu'
) -> Tuple[int, float]:
    """
    Load a training checkpoint and return the next epoch and best loss.

    Args:
        path (str): File path to load the checkpoint from.
        model (torch.nn.Module): Model into which to load weights.
        optimizer (optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        scaler (optional): GradScaler to load state into.
        device (str): Map location for torch.load.

    Returns:
        next_epoch (int): Epoch number to resume from.
        best_loss (float): Best validation loss recorded.
    """
    checkpoint = torch.load(path, map_location=device)
    epoch     = checkpoint.get('epoch', -1)
    best_loss = checkpoint.get('loss', np.inf)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # --- Modified AMP scaler loading ---
    if scaler is not None:
        ss = checkpoint.get('scaler_state_dict', None)
        if ss:
            scaler.load_state_dict(ss)
        else:
            logger.warning("No GradScaler state found in checkpoint; skipping AMP scaler load.")
    # -------------------------------------

    logger.info(f"Checkpoint loaded from epoch {epoch} with loss {best_loss:.4f}")
    return epoch + 1, best_loss
