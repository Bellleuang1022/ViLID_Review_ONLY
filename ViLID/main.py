import os
import argparse
import json
import random
from datetime import datetime
import logging

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, CLIPProcessor

from dataloader import get_dataloader, read_data_file
from trainer import train_model, validate
from model import ViLID, ViLIDLoss
from utility import save_checkpoint, load_checkpoint

# Logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"main_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser("ViLID training/evaluation")
    parser.add_argument("--data_file", type=str,
                        help="Single file to split (overrides --train_data/--val_data/--test_data)")
    parser.add_argument("--train_data", type=str, help="Path to pre-split train file")
    parser.add_argument("--val_data",   type=str, help="Path to pre-split val file")
    parser.add_argument("--test_data",  type=str, help="Path to pre-split test file")
    parser.add_argument("--text_encoder",  type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--image_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--hidden_size",       type=int,   default=256)
    parser.add_argument("--num_fusion_layers", type=int,   default=4)
    parser.add_argument("--num_fusion_heads",  type=int,   default=8)
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--freeze_encoders",   action="store_true")
    parser.add_argument("--gamma",   type=float, default=0.1)
    parser.add_argument("--beta",    type=float, default=5.0)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stop",   type=int,   default=5)
    parser.add_argument("--use_amp",      action="store_true")
    parser.add_argument("--clip_grad",    type=float, default=1.0)
    parser.add_argument("--use_image_cache",   action="store_true")
    parser.add_argument("--image_cache_dir",   type=str, default=".image_cache")
    parser.add_argument("--image_cache_size_gb", type=float, default=2.0)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--output_dir",    type=str, default="./output")
    parser.add_argument("--checkpoint_dir",type=str, default="./checkpoints")
    parser.add_argument("--save_every",    type=int, default=1)
    parser.add_argument("--resume",        type=str)
    parser.add_argument("--mode", choices=["train","test"], default="train")
    parser.add_argument("--eval_model",    type=str,
                        help="Path to checkpoint for evaluation (required in eval mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--image_col", type=str, default="image_url")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--text_rationale_col", type=str, default="text_rationale")
    parser.add_argument("--image_rationale_col", type=str, default="image_rationale")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--image_store_way", type=str, default="URL")
    parser.add_argument("--image_dir", type=str, default="images", required=False)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--data_name", type=str, default="Fakeddit")
    return parser.parse_args()

def build_config(args):
    return {
        "data_file": args.data_file,
        "train_data_path": args.train_data,
        "val_data_path":   args.val_data,
        "test_data_path":  args.test_data,
        "text_encoder_name":  args.text_encoder,
        "image_encoder_name": args.image_encoder,
        "hidden_size":        args.hidden_size,
        "num_fusion_layers":  args.num_fusion_layers,
        "num_fusion_heads":   args.num_fusion_heads,
        "dropout":            args.dropout,
        "freeze_encoders":    args.freeze_encoders,
        "gamma":              args.gamma,
        "beta":               args.beta,
        "batch_size":         args.batch_size,
        "num_epochs":         args.epochs,
        "learning_rate":      args.lr,
        "encoder_lr":         args.encoder_lr,
        "weight_decay":       args.weight_decay,
        "patience":           args.early_stop,
        "use_amp":            args.use_amp,
        "clip_grad_norm":     args.clip_grad,
        "use_image_cache":    args.use_image_cache,
        "image_cache_dir":    args.image_cache_dir,
        "image_cache_size_gb":args.image_cache_size_gb,
        "num_workers":        args.num_workers,
        "output_dir":         args.output_dir,
        "checkpoint_dir":     args.checkpoint_dir,
        "save_every":         args.save_every,
        "resume_checkpoint":  args.resume,
        "mode":               args.mode,
        "eval_model":         args.eval_model,
        "seed":              args.seed,
        "text_col":           args.text_col,
        "image_col":          args.image_col,
        "label_col":          args.label_col,
        "text_rationale_col": args.text_rationale_col,
        "image_rationale_col":args.image_rationale_col,
        "id_col":             args.id_col,
        "image_store_way":    args.image_store_way,
        "image_dir":          args.image_dir,
        "use_lr_scheduler":   args.use_lr_scheduler,
        "data_name":          args.data_name,
    }

def maybe_split_single_file(cfg):
    single = cfg["data_file"]
    if not single:
        return

    df = read_data_file(single)
    if "label" not in df.columns:
        raise ValueError("Need a 'label' column for stratified split")

    split_dir = os.path.join(cfg["output_dir"], "splits")
    os.makedirs(split_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(single))[0]
    seed = cfg["seed"]
    data_id = f"{cfg['data_name']}_{base_name}_seed{seed}"

    train_csv = os.path.join(split_dir, f"train_{data_id}.csv")
    val_csv   = os.path.join(split_dir, f"val_{data_id}.csv")
    test_csv  = os.path.join(split_dir, f"test_{data_id}.csv")

    if all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        logger.info(f"Split files already exist for {data_id}; skipping split.")
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=cfg["seed"],
            stratify=df["label"].astype(int)
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=cfg["seed"],
            stratify=temp_df["label"].astype(int)
        )
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        logger.info(f"Split single file into: {train_csv}, {val_csv}, {test_csv}")

    cfg.update({
        "train_data_path": train_csv,
        "val_data_path":   val_csv,
        "test_data_path":  test_csv,
    })

def main():
    args = parse_args()
    cfg  = build_config(args)
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    logger.info("Configuration:\n" + json.dumps(cfg, indent=2))

    maybe_split_single_file(cfg)

    if cfg["mode"] == "eval" and not cfg["eval_model"]:
        raise ValueError("--eval_model is required when --mode eval")

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder_name"])
    image_processor = CLIPProcessor.from_pretrained(cfg["image_encoder_name"]).image_processor

    if cfg["mode"] == "train":
        train_model(cfg, tokenizer, image_processor)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ViLID(
            text_model_name=cfg["text_encoder_name"],
            image_model_name=cfg["image_encoder_name"],
            hidden_size=cfg["hidden_size"],
            num_fusion_layers=cfg["num_fusion_layers"],
            num_fusion_heads=cfg["num_fusion_heads"],
            dropout=cfg["dropout"],
            gamma=cfg["gamma"],
            beta=cfg["beta"],
            freeze_encoders=cfg["freeze_encoders"],
        )
        ckpt = torch.load(cfg["eval_model"], map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.to(device)

        test_loader, _ = get_dataloader(
            data_path=cfg["test_data_path"],
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            use_image_cache=cfg["use_image_cache"],
            image_cache_dir=cfg["image_cache_dir"],
            image_cache_size_gb=cfg["image_cache_size_gb"],
        )
        criterion = ViLIDLoss(gamma=cfg["gamma"], lambda_weight_decay=cfg["weight_decay"])
        metrics = validate(model, test_loader, criterion, device, use_amp=cfg["use_amp"])

        eval_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(cfg["output_dir"], f"eval_results_{eval_time}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation results:\n" + json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
