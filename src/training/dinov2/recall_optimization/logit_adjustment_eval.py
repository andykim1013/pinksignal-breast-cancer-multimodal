import os
import sys
import json
import argparse
import logging

import yaml
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


LABEL_MAP = {
    "benign": 0,
    "malignant": 1,
}


def setup_project_path(root):
    if root not in sys.path:
        sys.path.append(root)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state_dict_robust(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    return model


def get_class_prior_from_train(csv_path, eps=1e-8):
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].copy()

    if len(train_df) == 0:
        raise ValueError("No train rows found in CSV.")

    labels = train_df["label"].map(LABEL_MAP).values
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    priors = counts / counts.sum()
    priors = np.clip(priors, eps, 1.0)

    return priors, counts


def collect_logits(model, loader, device):
    model.eval()

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for mammo_batch, us_batch, labels in loader:
            mammo_batch = mammo_batch.to(device)
            us_batch = us_batch.to(device)
            labels = labels.to(device)

            logits = model(mammo_batch, us_batch)

            all_logits.append(logits.detach().cpu())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = np.array(all_labels)

    return all_labels, all_logits


def safe_auc(labels, scores):
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def compute_metrics_from_preds(labels, preds, scores):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "roc_auc": safe_auc(labels, scores),
        "benign_precision": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "malignant_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def evaluate_argmax(labels, logits):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    scores = probs[:, 1]
    preds = np.argmax(logits, axis=1)

    metrics = compute_metrics_from_preds(labels, preds, scores)
    report = classification_report(
        labels,
        preds,
        target_names=["benign", "malignant"],
        zero_division=0,
    )

    return metrics, report, preds, scores


def evaluate_threshold(labels, logits, threshold):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    scores = probs[:, 1]
    preds = (scores >= threshold).astype(int)

    metrics = compute_metrics_from_preds(labels, preds, scores)
    report = classification_report(
        labels,
        preds,
        target_names=["benign", "malignant"],
        zero_division=0,
    )

    return metrics, report, preds, scores


def sweep_thresholds(labels, logits):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    scores = probs[:, 1]

    rows = []

    for threshold in np.linspace(0.01, 0.99, 99):
        preds = (scores >= threshold).astype(int)
        metrics = compute_metrics_from_preds(labels, preds, scores)

        rows.append({
            "threshold": float(threshold),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_roc_auc": metrics["roc_auc"],
            "val_malignant_precision": metrics["malignant_precision"],
            "val_malignant_recall": metrics["malignant_recall"],
            "val_specificity": metrics["specificity"],
            "val_sensitivity": metrics["sensitivity"],
        })

    df = pd.DataFrame(rows)

    # 목적: accuracy가 아니라 macro F1 + malignant recall 중심
    df = df.sort_values(
        by=["val_macro_f1", "val_malignant_recall", "val_roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def apply_logit_adjustment(logits, class_priors, tau):
    """
    Menon et al. post-hoc logit adjustment:
    adjusted_logit_y = logit_y - tau * log(pi_y)

    pi_y: class prior
    tau: adjustment strength
    """
    log_priors = np.log(class_priors)
    adjusted_logits = logits - tau * log_priors.reshape(1, -1)
    return adjusted_logits


def sweep_logit_adjustment(labels, logits, class_priors):
    rows = []

    for tau in np.linspace(0.0, 3.0, 31):
        adjusted_logits = apply_logit_adjustment(logits, class_priors, tau)
        probs = F.softmax(torch.tensor(adjusted_logits), dim=1).numpy()
        scores = probs[:, 1]
        preds = np.argmax(adjusted_logits, axis=1)

        metrics = compute_metrics_from_preds(labels, preds, scores)

        rows.append({
            "tau": float(tau),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_roc_auc": metrics["roc_auc"],
            "val_malignant_precision": metrics["malignant_precision"],
            "val_malignant_recall": metrics["malignant_recall"],
            "val_specificity": metrics["specificity"],
            "val_sensitivity": metrics["sensitivity"],
        })

    df = pd.DataFrame(rows)

    # 목적: accuracy보다 macro F1 + malignant recall 중심
    df = df.sort_values(
        by=["val_macro_f1", "val_malignant_recall", "val_roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def plot_confusion_matrix(cm, save_path, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["benign", "malignant"],
        yticklabels=["benign", "malignant"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(labels, scores_dict, save_path):
    plt.figure(figsize=(7, 7))

    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_value = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Threshold / Logit Adjustment ROC")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_predictions(output_dir, labels, base_scores, base_preds, threshold_scores, threshold_preds, logit_scores, logit_preds):
    pred_df = pd.DataFrame({
        "label": labels,
        "label_name": pd.Series(labels).map({0: "benign", 1: "malignant"}),

        "base_score_malignant": base_scores,
        "base_pred": base_preds,
        "base_pred_name": pd.Series(base_preds).map({0: "benign", 1: "malignant"}),

        "threshold_score_malignant": threshold_scores,
        "threshold_pred": threshold_preds,
        "threshold_pred_name": pd.Series(threshold_preds).map({0: "benign", 1: "malignant"}),

        "logit_adjusted_score_malignant": logit_scores,
        "logit_adjusted_pred": logit_preds,
        "logit_adjusted_pred_name": pd.Series(logit_preds).map({0: "benign", 1: "malignant"}),
    })

    save_path = os.path.join(output_dir, "threshold_logit_adjustment_test_predictions.csv")
    pred_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return save_path


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    setup_project_path(args.root)

    from src.training.dinov2.multimodal_model.multimodal_dataset import MultimodalDataset
    from src.training.dinov2.multimodal_model.multimodal_architecture import MultimodalFusionModel

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_yaml(args.config)
    with open(config["models"]["unimodal_config_path"], "r", encoding="utf-8") as f:
        unimodal_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Output dir: {args.output_dir}")

    class_priors, class_counts = get_class_prior_from_train(config["data"]["csv_path"])
    logging.info(f"Train class counts [benign, malignant]: {class_counts.tolist()}")
    logging.info(f"Train class priors [benign, malignant]: {class_priors.tolist()}")

    val_dataset = MultimodalDataset(
        csv_file=config["data"]["csv_path"],
        split="validation",
        image_size=config["training"]["image_size"],
    )
    test_dataset = MultimodalDataset(
        csv_file=config["data"]["csv_path"],
        split="test",
        image_size=config["training"]["image_size"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = MultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config["models"]["us_model_path"],
        mammo_model_path=config["models"]["mammo_model_path"],
    ).to(device)

    model = load_state_dict_robust(model, args.model_path, device)
    model.eval()

    logging.info("Collecting validation logits...")
    y_val, val_logits = collect_logits(model, val_loader, device)

    logging.info("Collecting test logits...")
    y_test, test_logits = collect_logits(model, test_loader, device)

    # 1) Original argmax baseline
    base_metrics, base_report, base_preds, base_scores = evaluate_argmax(y_test, test_logits)

    # 2) Threshold tuning on validation
    threshold_df = sweep_thresholds(y_val, val_logits)
    threshold_csv = os.path.join(args.output_dir, "validation_threshold_sweep.csv")
    threshold_df.to_csv(threshold_csv, index=False, encoding="utf-8-sig")

    best_threshold = float(threshold_df.iloc[0]["threshold"])
    threshold_metrics, threshold_report, threshold_preds, threshold_scores = evaluate_threshold(
        y_test,
        test_logits,
        best_threshold,
    )

    # 3) Post-hoc logit adjustment on validation
    logit_df = sweep_logit_adjustment(y_val, val_logits, class_priors)
    logit_csv = os.path.join(args.output_dir, "validation_logit_adjustment_sweep.csv")
    logit_df.to_csv(logit_csv, index=False, encoding="utf-8-sig")

    best_tau = float(logit_df.iloc[0]["tau"])
    test_adjusted_logits = apply_logit_adjustment(test_logits, class_priors, best_tau)
    logit_metrics, logit_report, logit_preds, logit_scores = evaluate_argmax(y_test, test_adjusted_logits)

    # Save prediction table
    pred_path = save_predictions(
        args.output_dir,
        y_test,
        base_scores,
        base_preds,
        threshold_scores,
        threshold_preds,
        logit_scores,
        logit_preds,
    )

    # Save summary
    summary = {
        "method": "Recall-oriented optimization: threshold tuning and post-hoc logit adjustment",
        "config": args.config,
        "model_path": args.model_path,
        "class_counts_train": {
            "benign": int(class_counts[0]),
            "malignant": int(class_counts[1]),
        },
        "class_priors_train": {
            "benign": float(class_priors[0]),
            "malignant": float(class_priors[1]),
        },
        "original_argmax": {
            "threshold": "argmax",
            "test_metrics": base_metrics,
        },
        "threshold_tuning": {
            "selected_by": "validation macro_f1, malignant_recall, roc_auc",
            "best_threshold": best_threshold,
            "validation_row": threshold_df.iloc[0].to_dict(),
            "test_metrics": threshold_metrics,
        },
        "posthoc_logit_adjustment": {
            "formula": "adjusted_logits = logits - tau * log(class_prior)",
            "selected_by": "validation macro_f1, malignant_recall, roc_auc",
            "best_tau": best_tau,
            "validation_row": logit_df.iloc[0].to_dict(),
            "test_metrics": logit_metrics,
        },
        "paths": {
            "threshold_sweep": threshold_csv,
            "logit_adjustment_sweep": logit_csv,
            "predictions": pred_path,
        },
    }

    summary_path = os.path.join(args.output_dir, "threshold_logit_adjustment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save reports
    report_txt = (
        "=" * 80 + "\n"
        "Recall-oriented Optimization: Threshold Tuning / Logit Adjustment\n"
        "=" * 80 + "\n\n"
        "[Original MMIBC Argmax]\n"
        f"{json.dumps(base_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{base_report}\n\n"
        "[Validation Threshold Tuning]\n"
        f"best_threshold = {best_threshold:.6f}\n"
        f"{json.dumps(threshold_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{threshold_report}\n\n"
        "[Post-hoc Logit Adjustment]\n"
        f"best_tau = {best_tau:.6f}\n"
        f"{json.dumps(logit_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{logit_report}\n"
    )

    report_path = os.path.join(args.output_dir, "threshold_logit_adjustment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # Save plots
    plot_confusion_matrix(
        np.array(base_metrics["confusion_matrix"]),
        os.path.join(args.output_dir, "cm_original_argmax.png"),
        "Original MMIBC Argmax",
    )

    plot_confusion_matrix(
        np.array(threshold_metrics["confusion_matrix"]),
        os.path.join(args.output_dir, "cm_threshold_tuning.png"),
        f"Threshold Tuning - threshold={best_threshold:.3f}",
    )

    plot_confusion_matrix(
        np.array(logit_metrics["confusion_matrix"]),
        os.path.join(args.output_dir, "cm_logit_adjustment.png"),
        f"Post-hoc Logit Adjustment - tau={best_tau:.2f}",
    )

    plot_roc(
        y_test,
        {
            "Original": base_scores,
            "Threshold tuning": threshold_scores,
            "Logit adjustment": logit_scores,
        },
        os.path.join(args.output_dir, "threshold_logit_adjustment_roc.png"),
    )

    logging.info("Done.")
    logging.info(f"Saved report: {report_path}")
    logging.info(f"Saved summary: {summary_path}")
    logging.info(f"Saved threshold sweep: {threshold_csv}")
    logging.info(f"Saved logit adjustment sweep: {logit_csv}")
    logging.info(f"Saved predictions: {pred_path}")

    print("\n" + "=" * 80)
    print("ORIGINAL MMIBC ARGMAX")
    print("=" * 80)
    print(base_report)

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING")
    print("=" * 80)
    print(f"best_threshold = {best_threshold:.6f}")
    print(threshold_report)

    print("\n" + "=" * 80)
    print("POST-HOC LOGIT ADJUSTMENT")
    print("=" * 80)
    print(f"best_tau = {best_tau:.6f}")
    print(logit_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recall-oriented threshold tuning and post-hoc logit adjustment for MMIBC."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root path.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/training/dinov2/multimodal_model/config_no_mammo_leak.yaml",
        help="Path to clean multimodal config.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/best_multimodal_model.pth",
        help="Path to trained MMIBC multimodal model.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/recall_optimization/threshold_logit_adjustment_clean",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    args = parser.parse_args()
    main(args)