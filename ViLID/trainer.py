import os
import time
import random
import logging
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from model import ViLID, ViLIDLoss
from dataloader import get_dataloader
from utility import EarlyStopping, save_checkpoint, load_checkpoint

# Logging setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"trainer_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ViLIDLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    scheduler=None,
    clip_grad_norm: float = None,
    use_amp: bool = False
) -> dict:
    model.train()
    total_loss = total_bce = total_align = total_wd = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Train"):
        optimizer.zero_grad()
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        texts    = [m['text_raw'] for m in batch['metadata']]
        text_rats= [m['text_rationale_raw'] for m in batch['metadata']]
        img_rats = [m['image_rationale_raw'] for m in batch['metadata']]

        with autocast(enabled=use_amp):
            outputs   = model(texts, images, text_rats, img_rats)
            loss_dict = criterion(
                logits=outputs['logits'],
                labels=labels,
                S_inc=outputs['S_inc'],
                S_r=outputs['S_r'],
                params=list(model.classifier.parameters())
            )
            loss = loss_dict['total']

        if use_amp and scaler:
            scaler.scale(loss).backward()
            if clip_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        bsz = labels.size(0)
        total_loss  += loss.item() * bsz
        total_bce   += loss_dict['bce'].item() * bsz
        total_align += loss_dict['align'].item() * bsz
        total_wd    += loss_dict.get('wd', 0.0) * bsz
        preds = (outputs['y_pred'] > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return {
        'loss': total_loss / n,
        'bce_loss': total_bce / n,
        'align_loss': total_align / n,
        'wd_loss': total_wd / n,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ViLIDLoss,
    device: torch.device,
    use_amp: bool = False
) -> dict:
    model.eval()
    total_loss = total_bce = total_align = total_wd = 0.0
    all_preds, all_probs, all_labels = [], [], []
    all_S_inc, all_S_r = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val  "):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            texts    = [m['text_raw'] for m in batch['metadata']]
            text_rats= [m['text_rationale_raw'] for m in batch['metadata']]
            img_rats = [m['image_rationale_raw'] for m in batch['metadata']]

            with autocast(enabled=use_amp):
                outputs   = model(texts, images, text_rats, img_rats)
                loss_dict = criterion(
                    logits=outputs['logits'],
                    labels=labels,
                    S_inc=outputs['S_inc'],
                    S_r=outputs['S_r'],
                    params=list(model.classifier.parameters())
                )
            bsz = labels.size(0)
            total_loss  += loss_dict['total'].item() * bsz
            total_bce   += loss_dict['bce'].item()   * bsz
            total_align += loss_dict['align'].item() * bsz
            total_wd    += loss_dict.get('wd', 0.0) * bsz

            probs = outputs['y_pred'].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_S_inc.extend(outputs['S_inc'].cpu().numpy().tolist())
            all_S_r.extend(outputs['S_r'].cpu().numpy().tolist())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    return {
        'loss': total_loss / n,
        'bce_loss': total_bce / n,
        'align_loss': total_align / n,
        'wd_loss': total_wd / n,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'avg_S_inc': np.mean(all_S_inc),
        'avg_S_r': np.mean(all_S_r)
    }

def train_model(config: dict, tokenizer=None, image_processor=None, resume_checkpoint=None) -> ViLID:
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data (unchanged)
    train_loader, _ = get_dataloader(
        data_path=config['train_data_path'],
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        max_text_length=512,
        max_rationale_length=512,
        text_col=config['text_col'],
        image_col=config['image_col'],
        label_col=config['label_col'],
        text_rationale_col=config['text_rationale_col'],
        image_rationale_col=config['image_rationale_col'],
        id_col=config['id_col'],
        image_store_way=config['image_store_way'],
        image_dir=config['image_dir']
    )
    val_loader, _ = get_dataloader(
        data_path=config['val_data_path'],
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        max_text_length=512,
        max_rationale_length=512,
        text_col=config['text_col'],
        image_col=config['image_col'],
        label_col=config['label_col'],
        text_rationale_col=config['text_rationale_col'],
        image_rationale_col=config['image_rationale_col'],
        id_col=config['id_col'],
        image_store_way=config['image_store_way'],
        image_dir=config['image_dir']
    )

    # Model
    model = ViLID(
        text_model_name=config['text_encoder_name'],
        image_model_name=config['image_encoder_name'],
        hidden_size=config['hidden_size'],
        num_fusion_layers=config['num_fusion_layers'],
        num_fusion_heads=config['num_fusion_heads'],
        dropout=config['dropout'],
        gamma=config['gamma'],
        beta=config['beta'],
        freeze_encoders=config['freeze_encoders']
    ).to(device)

    # Optimizer (no LR scheduler)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = None

    # Criterion & AMP
    criterion = ViLIDLoss(gamma=config['gamma'], weight_decay=config['weight_decay'])
    scaler = GradScaler() if config.get('use_amp') else None

    # Early stopping
    early = EarlyStopping(patience=config['patience'], verbose=True)

    # ────────────────────────────
    # Resume / checkpoint naming using data_name only
    # ────────────────────────────
    ckpt_dir = config.get('save_check_dir', config['checkpoint_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)
    pattern = f"*seed{config['seed']}_data{config['data_name']}_*.pt"
    matches = glob.glob(os.path.join(ckpt_dir, pattern))

    if matches:
        checkpoint_path = max(matches, key=os.path.getctime)
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device=device
        )
        logger.info(f"Resuming from checkpoint {checkpoint_path} (epoch {start_epoch}, loss {best_loss:.4f})")
        epoch_iter = range(start_epoch, config['num_epochs'])
    else:
        ts = datetime.now().strftime('%Y%m%d%H%M')
        fname = f"seed{config['seed']}_data{config['data_name']}_{ts}.pt"
        checkpoint_path = os.path.join(ckpt_dir, fname)
        start_epoch, best_loss = 0, float('inf')
        epoch_iter = range(config['num_epochs'])
        logger.info(f"No existing checkpoint; will save to {checkpoint_path}")

    # ────────────────────────────
    # Training loop
    # ────────────────────────────
    for epoch in epoch_iter:
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        train_m = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, scheduler, config.get('clip_grad'), config.get('use_amp', False)
        )
        val_m = validate(model, val_loader, criterion, device, config.get('use_amp', False))

        logger.info(f"Train: {train_m} | Val: {val_m}")

        if val_m['loss'] < best_loss:
            best_loss = val_m['loss']
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=best_loss,
                path=checkpoint_path,
                scheduler=scheduler,
                scaler=scaler
            )

        if early(val_m['loss'], model):
            break

    return model
