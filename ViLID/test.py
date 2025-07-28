#!/usr/bin/env python
import os
import argparse
import json
import torch
import numpy as np # Import numpy for type checking
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, classification_report
)
from transformers import AutoTokenizer, CLIPProcessor
import logging
from pathlib import Path
import warnings

from model import ViLID
from dataloader import get_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser("Test ViLID")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the checkpoint (.pt file) of the trained model")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to a JSON config file (used if config not found in checkpoint)")
    p.add_argument("--test_data", type=str, required=True,
                   help="Path to the test data file (CSV, TSV, JSONL)")
    p.add_argument("--output_dir", type=str, default="./test_results",
                   help="Directory to save evaluation results and plots")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for testing")
    p.add_argument("--gpu_id", type=str, default="0",
                   help="Single GPU ID to use (e.g., '0'). Leave empty or set to '-' for CPU.")
    p.add_argument("--use_amp", action="store_true",
                   help="Enable Automatic Mixed Precision (AMP) for inference")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary decision threshold for predictions")
    p.add_argument("--save_preds", action="store_true",
                   help="Save per-sample predictions to a JSON file")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of worker processes for data loading")
    p.add_argument("--use_image_cache", action="store_true",
                   help="Enable image caching")
    p.add_argument("--image_cache_dir", type=str, default=".image_cache",
                   help="Directory for image cache")
    p.add_argument("--image_cache_size_gb", type=float, default=2.0,
                   help="Maximum size of the image cache in GB")
    p.add_argument("--image_dir", type=str, default=None,
                   help="Directory for local images if image_store_way is 'LOCAL'. Overrides config if set.")
    return p.parse_args()

def load_config_from_file(path):
    logger.info(f"Loading configuration from file: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r') as f:
        cfg = json.load(f)
    logger.info("Configuration loaded successfully from file.")
    return cfg

def load_model_and_config(model_path: str, config_path: str | None, device: torch.device):
    logger.info(f"Loading checkpoint from: {model_path}")
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        logger.info(f"Checkpoint loaded successfully (weights_only=True). Keys: {list(ckpt.keys())}")
    except Exception as e_true:
        logger.warning(f"Failed loading checkpoint with weights_only=True: {e_true}. Trying weights_only=False.")
        logger.warning("Loading with weights_only=False is potentially insecure if the checkpoint source is untrusted.")
        try:
            warnings.warn(
                "You are using `torch.load` with `weights_only=False`, which uses the default pickle module implicitly. "
                "It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. "
                "See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details. "
                "We recommend setting `weights_only=True` wherever possible.",
                FutureWarning
            )
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(f"Checkpoint loaded successfully (weights_only=False). Keys: {list(ckpt.keys())}")
        except Exception as e_false:
            logger.error(f"Failed loading checkpoint even with weights_only=False: {e_false}")
            raise e_false

    cfg = None
    if 'config' in ckpt:
        cfg = ckpt['config']
        logger.info("Configuration successfully loaded from checkpoint.")
    else:
        logger.warning("Configuration ('config' key) not found in checkpoint.")
        if config_path:
            logger.info(f"Attempting to load configuration from provided file: {config_path}")
            try:
                cfg = load_config_from_file(config_path)
            except Exception as e:
                logger.error(f"Failed to load configuration from file '{config_path}': {e}")
                raise ValueError(
                    "Config not found in checkpoint AND failed to load from "
                    f"provided --config_path '{config_path}'. Cannot proceed."
                ) from e
        else:
            logger.error("Config not found in checkpoint and no --config_path provided.")
            raise ValueError(
                "Configuration not found in checkpoint and --config_path was not specified. "
                "Please provide a configuration file using --config_path or ensure the "
                "checkpoint was saved with the 'config' key."
            )

    if cfg is None:
         raise RuntimeError("Configuration could not be loaded. Cannot instantiate model.")

    log_cfg_details = {k: cfg.get(k) for k in [
        "text_encoder_name", "image_encoder_name", "hidden_size",
        "num_fusion_layers", "num_fusion_heads", "dropout", "gamma", "beta",
        "freeze_encoders"
    ]}
    logger.info(f"Using Model Config: {json.dumps(log_cfg_details, indent=2)}")

    model = ViLID(
        text_model_name=cfg["text_encoder_name"],
        image_model_name=cfg["image_encoder_name"],
        hidden_size=cfg["hidden_size"],
        num_fusion_layers=cfg["num_fusion_layers"],
        num_fusion_heads=cfg["num_fusion_heads"],
        dropout=cfg.get("dropout", 0.1),
        gamma=cfg.get("gamma", 0.1),
        beta=cfg.get("beta", 5.0),
        freeze_encoders=cfg.get("freeze_encoders", True)
    )

    model_state_dict_key = 'model_state_dict'
    if model_state_dict_key not in ckpt:
        logger.warning(f"'{model_state_dict_key}' not found in checkpoint. Checking if checkpoint is the state_dict itself.")
        # Check if the ckpt itself is a state_dict (e.g. from an older save format)
        # A simple heuristic: check if all keys in ckpt are also in an initialized model's state_dict
        temp_model_keys = set(model.state_dict().keys())
        if all(k in temp_model_keys for k in ckpt.keys()):
            logger.info("Checkpoint appears to be a raw state_dict. Attempting to load directly.")
            model_state_dict = ckpt
        else:
            logger.error(f"Cannot determine model_state_dict. Checkpoint keys: {list(ckpt.keys())}")
            raise ValueError(f"'{model_state_dict_key}' not found in checkpoint and checkpoint is not a raw state_dict.")
    else:
        model_state_dict = ckpt[model_state_dict_key]

    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        logger.error(f"Error loading state_dict: {e}")
        logger.error("This might be due to a mismatch between the model architecture defined here and the one saved in the checkpoint.")
        logger.error("Ensure the configuration (hidden_size, layers, heads, encoders, etc.) matches the trained model.")
        raise e

    model.to(device)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model, cfg

def evaluate(model, loader, device, threshold, use_amp, save_preds):
    all_true_labels, all_pred_labels, all_pred_probs = [], [], []
    all_S_inc_scores, all_S_r_scores = [], []
    predictions_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Testing")):
            try:
                raw_texts = [meta['text_raw'] for meta in batch['metadata']]
                raw_text_rationales = [meta['text_rationale_raw'] for meta in batch['metadata']]
                raw_image_rationales = [meta['image_rationale_raw'] for meta in batch['metadata']]
            except KeyError as e:
                logger.error(f"Batch {batch_idx}: Missing expected key in batch['metadata']: {e}")
                continue

            images_tensor = batch["images"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                try:
                    out = model(
                        texts=raw_texts,
                        images=images_tensor,
                        text_rationales=raw_text_rationales,
                        image_rationales=raw_image_rationales
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_idx}: Error during model forward pass: {e}")
                    raise e

            if "y_pred" not in out:
                 logger.error(f"Batch {batch_idx}: Model output missing 'y_pred' key. Output keys: {list(out.keys())}")
                 continue # Skip batch if output is malformed

            probs_tensor = out["y_pred"]
            probs = probs_tensor.cpu().numpy().squeeze()

            current_batch_size = len(labels)
            if probs.ndim == 0 and current_batch_size == 1:
                probs = np.array([probs.item()])
            elif probs.shape == () and current_batch_size == 1: # Squeezed to a 0-d array
                probs = np.array([probs.item()])
            elif probs.ndim > 0 and probs.shape[0] != current_batch_size:
                 logger.warning(f"Batch {batch_idx}: Probability shape {probs.shape} mismatch with batch size {current_batch_size}. Squeezing.")
                 probs = probs.squeeze()
                 if probs.ndim > 0 and probs.shape[0] != current_batch_size:
                     logger.error(f"Batch {batch_idx}: Probability shape mismatch persists: {probs.shape}. Skipping batch.")
                     continue
                 elif probs.ndim == 0 and current_batch_size == 1: # Reshaped to scalar after squeeze
                     probs = np.array([probs.item()])
                 elif probs.ndim == 0 and current_batch_size > 1: # Error case
                     logger.error(f"Batch {batch_idx}: Probability squeezed to scalar for batch size > 1. Skipping batch.")
                     continue


            predicted_labels = (probs >= threshold).astype(int) # NumPy int, will be converted later

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted_labels) # Extends with NumPy ints
            all_pred_probs.extend(probs) # Extends with NumPy floats

            if "S_inc" in out:
                s_inc_vals = out["S_inc"].cpu().numpy().squeeze()
                if s_inc_vals.ndim == 0 and current_batch_size == 1: s_inc_vals = np.array([s_inc_vals.item()])
                if s_inc_vals.shape == (current_batch_size,): all_S_inc_scores.extend(s_inc_vals)
                else: logger.warning(f"Batch {batch_idx}: Shape mismatch for S_inc: {s_inc_vals.shape}")

            if "S_r" in out:
                s_r_vals = out["S_r"].cpu().numpy().squeeze()
                if s_r_vals.ndim == 0 and current_batch_size == 1: s_r_vals = np.array([s_r_vals.item()])
                if s_r_vals.shape == (current_batch_size,): all_S_r_scores.extend(s_r_vals)
                else: logger.warning(f"Batch {batch_idx}: Shape mismatch for S_r: {s_r_vals.shape}")

            if save_preds:
                for i in range(current_batch_size):
                    meta_item = batch["metadata"][i]
                    # --- FIX: Cast id to str ---
                    pred_item_id = str(meta_item.get("id", f"item_batch{batch_idx}_{i}"))
                    # --- End FIX ---
                    pred_item = {
                        "id": pred_item_id,
                        "true_label": int(labels[i].cpu().item()),
                        "predicted_label": int(predicted_labels[i]), # np.intX converted to Python int
                        "probability_class1": float(probs[i]),     # np.floatX converted to Python float
                    }
                    if "S_inc" in out and i < len(out["S_inc"]):
                         pred_item["S_inc"] = float(out["S_inc"][i].cpu().item())
                    if "S_r" in out and i < len(out["S_r"]):
                         pred_item["S_r"] = float(out["S_r"][i].cpu().item())
                    predictions_list.append(pred_item)

    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels) # Array of NumPy ints
    y_prob = np.array(all_pred_probs) # Array of NumPy floats

    if len(y_true) == 0:
        logger.error("No valid predictions collected. Cannot calculate metrics.")
        return {}, [], ([], [], 0.0)

    if len(np.unique(y_true)) < 2:
        logger.warning("ROC AUC score is not defined when only one class is present in true labels.")
        fpr, tpr, roc_auc_score = np.array([0.0]), np.array([0.0]), 0.0
    elif len(np.unique(y_prob)) < 2:
        logger.warning(f"Only {len(np.unique(y_prob))} unique probability value(s) found. ROC AUC might be ill-defined or 0.5.")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_score = auc(fpr, tpr) if len(fpr) > 1 and len(tpr) > 1 else 0.5
    else:
         fpr, tpr, _ = roc_curve(y_true, y_prob)
         roc_auc_score = auc(fpr, tpr)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist() # Already Python list of lists of Python ints
    report_dict = classification_report(
        y_true, y_pred, target_names=["Class 0", "Class 1"], output_dict=True, zero_division=0
    )
    # report_dict will be sanitized later by convert_numpy_types

    report_str = classification_report(
        y_true, y_pred, target_names=["Class 0", "Class 1"], zero_division=0
    )
    logger.info(f"Classification Report:\n{report_str}")

    metrics = {
        "accuracy": acc, # np.float64
        "precision": precision, # np.float64
        "recall": recall, # np.float64
        "f1_score": f1, # np.float64
        "roc_auc": roc_auc_score, # np.float64
        "confusion_matrix": cm, # Python list of lists of Python ints
        "classification_report_dict": report_dict, # Contains np types
        "threshold": threshold, # Python float
        "num_samples": len(y_true) # Python int
    }
    if all_S_inc_scores:
        metrics["avg_S_inc"] = np.mean(all_S_inc_scores) # np.float64
    if all_S_r_scores:
        metrics["avg_S_r"] = np.mean(all_S_r_scores) # np.float64

    roc_data_serializable = (fpr.tolist(), tpr.tolist(), roc_auc_score) # roc_auc_score is np.float64

    return metrics, predictions_list, roc_data_serializable

def plot_confusion_matrix(cm_data, output_path, class_names=None):
    import seaborn as sns
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm_data))]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(fpr_list, tpr_list, roc_auc_val, output_path):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc_val:.2f})') # roc_auc_val will be float after sanitizing
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")

# --- Helper function to convert NumPy types to Python native types for JSON serialization ---
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, np.integer): # Catches np.int32, np.int64, etc.
        return int(obj)
    elif isinstance(obj, np.floating): # Catches np.float32, np.float64, etc.
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist()) # Convert array to list and then sanitize list
    elif isinstance(obj, np.bool_): # Catches np.bool_
        return bool(obj)
    return obj
# --- End Helper function ---

def main():
    args = parse_args()

    if args.gpu_id and args.gpu_id != '-':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id.split(',')[0]}")
        else:
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_and_config(args.model_path, args.config_path, device)

    logger.info(f"Loading tokenizer: {cfg['text_encoder_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg['text_encoder_name'])
    logger.info(f"Loading image processor: {cfg['image_encoder_name']}")
    clip_processor = CLIPProcessor.from_pretrained(cfg['image_encoder_name'])
    image_processor = clip_processor.image_processor

    dataloader_image_dir = args.image_dir if args.image_dir is not None else cfg.get("image_dir")

    test_loader, _ = get_dataloader(
        data_path=args.test_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        text_col=cfg.get("text_col", "text"),
        image_col=cfg.get("image_col", "image_url"),
        label_col=cfg.get("label_col", "label"),
        text_rationale_col=cfg.get("text_rationale_col", "text_rationale"),
        image_rationale_col=cfg.get("image_rationale_col", "image_rationale"),
        id_col=cfg.get("id_col", "id"),
        image_store_way=cfg.get("image_store_way", "URL"),
        image_dir=dataloader_image_dir,
        use_image_cache=args.use_image_cache,
        image_cache_dir=args.image_cache_dir,
        image_cache_size_gb=args.image_cache_size_gb,
        max_text_length=cfg.get("max_text_length", 512),
        max_rationale_length=cfg.get("max_rationale_length", 512)
    )

    metrics, predictions, roc_data = evaluate(
        model,
        test_loader,
        device,
        threshold=args.threshold,
        use_amp=args.use_amp,
        save_preds=args.save_preds
    )

    # --- Sanitize metrics and roc_data before saving ---
    sanitized_metrics = convert_numpy_types(metrics)
    # roc_data is (fpr_list, tpr_list, roc_auc_score)
    # fpr_list and tpr_list are already Python lists from .tolist()
    # roc_auc_score needs to be Python float
    sanitized_roc_data = (roc_data[0], roc_data[1], float(roc_data[2]) if isinstance(roc_data[2], np.floating) else roc_data[2])


    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(sanitized_metrics, f, indent=4) # Save sanitized metrics
    logger.info(f"Evaluation metrics saved to {metrics_path}")

    # Plotting
    # Ensure confusion matrix data is native Python list of lists of ints
    cm_to_plot = sanitized_metrics.get("confusion_matrix", [])
    if cm_to_plot: # Check if confusion matrix exists and is not empty
        # Determine class names from sanitized classification report
        report_dict_to_plot = sanitized_metrics.get("classification_report_dict", {})
        # Extract class keys (e.g., "Class 0", "Class 1") excluding summary keys
        class_keys_from_report = [k for k in report_dict_to_plot.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

        if len(class_keys_from_report) == len(cm_to_plot):
            target_names = class_keys_from_report
        else:
            logger.warning("Mismatch between classification report keys and CM size for plotting. Using default names.")
            target_names = [f"Class {i}" for i in range(len(cm_to_plot))]

        plot_confusion_matrix(
            cm_to_plot,
            output_dir / "confusion_matrix.png",
            class_names=target_names
        )
    else:
        logger.warning("Confusion matrix data not found or empty in metrics. Skipping plot.")


    if sanitized_roc_data and len(sanitized_roc_data[0]) > 0 and len(sanitized_roc_data[1]) > 0:
        plot_roc_curve(
            sanitized_roc_data[0], sanitized_roc_data[1], sanitized_roc_data[2], # Use sanitized roc_auc
            output_dir / "roc_curve.png"
        )
    else:
        logger.warning("ROC data is invalid or empty, skipping ROC curve plot.")

    # Predictions list items are already sanitized individually for id, label, prob, S_inc, S_r
    if args.save_preds and predictions:
        preds_path = output_dir / "predictions.json"
        with open(preds_path, "w") as f:
            json.dump(predictions, f, indent=4) # 'predictions' should be fine now
        logger.info(f"Predictions saved to {preds_path}")

    logger.info("Testing complete.")
    logger.info(f"Results saved in {output_dir.resolve()}")

if __name__ == "__main__":
    main()
