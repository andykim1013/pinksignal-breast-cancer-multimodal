# src/training/dinov2/dual_roi_fusion/evaluate_dual_roi_validation.py
#
# Purpose:
#   Evaluate a trained Dual-ROI MMIBC model on the validation split.
#
# Use case:
#   ROI + class weight model 결과를 Clean Test가 아니라 Validation Result 기준으로 확인.
#
# Example:
#   python src\training\dinov2\dual_roi_fusion\evaluate_dual_roi_validation.py ^
#     --model_path "saved_models\best_multimodal_model_dual_roi_classweight2p0_bs4_seed42_init_mmibc.pth" ^
#     --output_dir "outputs\dual_roi_fusion\validation_eval_classweight2p0"

import os
import sys
import json
import argparse
import logging
import random
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. Basic utilities
# ============================================================

def setup_project_path(root):
    if root not in sys.path:
        sys.path.append(root)


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"validation_eval_{current_time}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    return log_file


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_state_dict_robust(model, model_path, device, strict=True):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=strict)
    return model


# ============================================================
# 2. Evaluation functions
# ============================================================

@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()

    all_labels = []
    all_logits = []

    for mammo_batch, us_batch, us_roi_batch, labels in loader:
        mammo_batch = mammo_batch.to(device)
        us_batch = us_batch.to(device)
        us_roi_batch = us_roi_batch.to(device)
        labels = labels.to(device)

        outputs = model(mammo_batch, us_batch, us_roi_batch)

        all_logits.append(outputs.detach().cpu())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = np.array(all_labels)

    return all_labels, all_logits


def compute_metrics(labels, logits):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    scores = probs[:, 1]
    preds = np.argmax(logits, axis=1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "benign_precision": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "benign_f1": float(f1_score(labels, preds, pos_label=0, zero_division=0)),
        "malignant_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    return metrics, preds, scores


def save_outputs(args, metrics, report, labels, preds, scores):
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "method": "Dual-ROI Validation Evaluation",
        "model_path": args.model_path,
        "config": args.config,
        "roi_csv": args.roi_csv,
        "roi_col": args.roi_col,
        "split": args.split,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "metrics": metrics,
    }

    summary_path = os.path.join(args.output_dir, f"{args.split}_validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(args.output_dir, f"{args.split}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DUAL-ROI VALIDATION RESULT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Split: {args.split}\n\n")
        f.write("[ADDITIONAL METRICS]\n")
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
        f.write("\n\n[CLASSIFICATION REPORT]\n")
        f.write(report)

    pred_df = pd.DataFrame({
        "label": labels,
        "label_name": pd.Series(labels).map({0: "benign", 1: "malignant"}),
        "score_malignant": scores,
        "pred": preds,
        "pred_name": pd.Series(preds).map({0: "benign", 1: "malignant"}),
    })

    pred_path = os.path.join(args.output_dir, f"{args.split}_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return summary_path, report_path, pred_path


# ============================================================
# 3. Main
# ============================================================

def main(args):
    setup_project_path(args.root)

    from src.training.dinov2.dual_roi_fusion.multimodal_dual_roi_dataset import MultimodalDualROIDataset
    from src.training.dinov2.dual_roi_fusion.dual_roi_architecture import DualROIMultimodalFusionModel

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    set_seed(args.seed)

    config = load_yaml(args.config)
    unimodal_config = load_yaml(config["models"]["unimodal_config_path"])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    logging.info("=" * 80)
    logging.info("Dual-ROI validation evaluation started")
    logging.info("=" * 80)
    logging.info(f"Device: {device}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"ROI CSV: {args.roi_csv}")
    logging.info(f"Split: {args.split}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Log file: {log_file}")

    dataset = MultimodalDualROIDataset(
        csv_file=args.roi_csv,
        split=args.split,
        image_size=config["training"]["image_size"],
        roi_col=args.roi_col,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = DualROIMultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config["models"]["us_model_path"],
        mammo_model_path=config["models"]["mammo_model_path"],
        dropout_rate=float(args.dropout_rate),
    ).to(device)

    model = load_state_dict_robust(model, args.model_path, device, strict=True)
    model.eval()

    # Evaluation-only mode. Encoders are not trained here.
    model.freeze_encoders = True

    labels, logits = collect_logits(model, loader, device)
    metrics, preds, scores = compute_metrics(labels, logits)

    report = classification_report(
        labels,
        preds,
        target_names=["benign", "malignant"],
        zero_division=0,
        digits=4,
    )

    summary_path, report_path, pred_path = save_outputs(
        args=args,
        metrics=metrics,
        report=report,
        labels=labels,
        preds=preds,
        scores=scores,
    )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION RESULT")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Split     : {args.split}")
    print()
    print(report)

    print("\n" + "=" * 80)
    print("ADDITIONAL VALIDATION METRICS")
    print("=" * 80)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Macro F1           : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1        : {metrics['weighted_f1']:.4f}")
    print(f"ROC AUC            : {metrics['roc_auc']:.4f}")
    print(f"Benign Precision   : {metrics['benign_precision']:.4f}")
    print(f"Benign Recall      : {metrics['benign_recall']:.4f}")
    print(f"Benign F1          : {metrics['benign_f1']:.4f}")
    print(f"Malignant Precision: {metrics['malignant_precision']:.4f}")
    print(f"Malignant Recall   : {metrics['malignant_recall']:.4f}")
    print(f"Malignant F1       : {metrics['malignant_f1']:.4f}")
    print(f"TN / FP / FN / TP  : {metrics['tn']} / {metrics['fp']} / {metrics['fn']} / {metrics['tp']}")

    print("\nSaved files:")
    print(f"Summary    : {summary_path}")
    print(f"Report     : {report_path}")
    print(f"Predictions: {pred_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Dual-ROI MMIBC model on validation split.")

    parser.add_argument(
        "--root",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/training/dinov2/multimodal_model/config_no_mammo_leak.yaml",
    )

    parser.add_argument(
        "--roi_csv",
        type=str,
        default="data/multimodal_pairs_roi_margin030_no_mammo_leak.csv",
    )

    parser.add_argument(
        "--roi_col",
        type=str,
        default="ultrasound_roi_path",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Trained Dual-ROI model path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dual_roi_fusion/validation_eval",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)