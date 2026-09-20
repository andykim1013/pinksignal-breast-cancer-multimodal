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
    roc_curve,
)

import matplotlib.pyplot as plt


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, f"logs_threshold_{current_time}")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "threshold_tuning.log")

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


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state_dict_robust(model, model_path, device, strict=True):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=strict)
    return model


def collect_scores(model, loader, device):
    model.eval()

    all_labels = []
    all_scores = []
    all_logits = []

    with torch.no_grad():
        for mammo_batch, us_batch, labels in loader:
            mammo_batch = mammo_batch.to(device)
            us_batch = us_batch.to(device)

            logits = model(mammo_batch, us_batch)
            probs = F.softmax(logits, dim=1)
            malignant_scores = probs[:, 1]

            all_logits.append(logits.detach().cpu())
            all_scores.extend(malignant_scores.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()

    return np.array(all_labels), np.array(all_scores), all_logits


def metrics_from_threshold(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "malignant_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "benign_precision": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    except Exception:
        metrics["roc_auc"] = None

    return metrics, preds


def sweep_thresholds(labels, scores, thresholds):
    rows = []

    for threshold in thresholds:
        m, _ = metrics_from_threshold(labels, scores, threshold)
        rows.append(m)

    return pd.DataFrame(rows)


def pick_thresholds(val_df, macro_tolerance=0.03, acc_tolerance=0.04):
    """
    Select three validation-based thresholds:

    1. best_macro_f1:
       - macro F1 maximum
       - tie-breaker: malignant recall

    2. best_malignant_f1:
       - malignant F1 maximum
       - tie-breaker: macro F1

    3. recall_constrained:
       - malignant recall maximum
       - but macro F1 and accuracy should not collapse too much compared with threshold 0.5
    """

    base_row = val_df.iloc[(val_df["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]
    base_macro = float(base_row["macro_f1"])
    base_acc = float(base_row["accuracy"])

    best_macro = (
        val_df.sort_values(
            ["macro_f1", "malignant_recall", "accuracy"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )

    best_malignant_f1 = (
        val_df.sort_values(
            ["malignant_f1", "macro_f1", "malignant_recall"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )

    constrained = val_df[
        (val_df["macro_f1"] >= base_macro - float(macro_tolerance))
        & (val_df["accuracy"] >= base_acc - float(acc_tolerance))
    ].copy()

    if len(constrained) == 0:
        recall_constrained = best_macro
    else:
        recall_constrained = (
            constrained.sort_values(
                ["malignant_recall", "macro_f1", "accuracy"],
                ascending=[False, False, False],
            )
            .iloc[0]
            .to_dict()
        )

    selected = {
        "baseline_0.50": base_row.to_dict(),
        "best_macro_f1": best_macro,
        "best_malignant_f1": best_malignant_f1,
        "recall_constrained": recall_constrained,
    }

    return selected


def plot_confusion_matrix(cm, save_path, title):
    cm = np.array(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["benign", "malignant"])
    plt.yticks(tick_marks, ["benign", "malignant"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(labels, scores, save_path, title):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_report(name, labels, scores, threshold, output_dir):
    metrics, preds = metrics_from_threshold(labels, scores, threshold)

    report = classification_report(
        labels,
        preds,
        target_names=["benign", "malignant"],
        zero_division=0,
    )

    report_path = os.path.join(output_dir, f"{name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"threshold: {threshold:.6f}\n\n")
        f.write("[metrics]\n")
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
        f.write("\n\n[classification_report]\n")
        f.write(report)

    cm_path = os.path.join(output_dir, f"{name}_confusion_matrix.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cm_path,
        f"{name} - threshold={threshold:.3f}",
    )

    return metrics, report, report_path


def main(args):
    setup_project_path(args.root)

    from src.training.dinov2.roi_fusion.multimodal_roi_dataset import MultimodalROIDataset
    from src.training.dinov2.multimodal_model.multimodal_architecture import MultimodalFusionModel

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    set_seed(args.seed)

    config = load_yaml(args.config)
    unimodal_config = load_yaml(config["models"]["unimodal_config_path"])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    logging.info("Starting ROI threshold tuning")
    logging.info(f"Device: {device}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"ROI CSV: {args.roi_csv}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Log file: {log_file}")

    val_dataset = MultimodalROIDataset(
        csv_file=args.roi_csv,
        split="validation",
        image_size=config["training"]["image_size"],
        roi_col=args.roi_col,
    )

    test_dataset = MultimodalROIDataset(
        csv_file=args.roi_csv,
        split="test",
        image_size=config["training"]["image_size"],
        roi_col=args.roi_col,
    )

    eval_generator = torch.Generator()
    eval_generator.manual_seed(args.seed)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    model = MultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config["models"]["us_model_path"],
        mammo_model_path=config["models"]["mammo_model_path"],
    ).to(device)

    model = load_state_dict_robust(model, args.model_path, device, strict=True)
    model.eval()

    y_val, val_scores, val_logits = collect_scores(model, val_loader, device)
    y_test, test_scores, test_logits = collect_scores(model, test_loader, device)

    thresholds = np.arange(args.min_threshold, args.max_threshold + 1e-9, args.step)
    val_sweep_df = sweep_thresholds(y_val, val_scores, thresholds)

    val_sweep_path = os.path.join(args.output_dir, "validation_threshold_sweep.csv")
    val_sweep_df.to_csv(val_sweep_path, index=False, encoding="utf-8-sig")

    selected = pick_thresholds(
        val_sweep_df,
        macro_tolerance=args.macro_tolerance,
        acc_tolerance=args.acc_tolerance,
    )

    selected_rows = []

    for name, row in selected.items():
        threshold = float(row["threshold"])

        test_metrics, test_report, report_path = save_report(
            name=f"test_{name}",
            labels=y_test,
            scores=test_scores,
            threshold=threshold,
            output_dir=args.output_dir,
        )

        selected_rows.append({
            "selection_name": name,
            "selected_threshold_from_val": threshold,
            "val_accuracy": float(row["accuracy"]),
            "val_macro_f1": float(row["macro_f1"]),
            "val_malignant_recall": float(row["malignant_recall"]),
            "val_malignant_f1": float(row["malignant_f1"]),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_malignant_precision": test_metrics["malignant_precision"],
            "test_malignant_recall": test_metrics["malignant_recall"],
            "test_malignant_f1": test_metrics["malignant_f1"],
            "test_specificity": test_metrics["specificity"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_tn": test_metrics["tn"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            "test_tp": test_metrics["tp"],
            "report_path": report_path,
        })

    selected_df = pd.DataFrame(selected_rows)
    selected_path = os.path.join(args.output_dir, "selected_threshold_test_results.csv")
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "label": y_test,
        "label_name": pd.Series(y_test).map({0: "benign", 1: "malignant"}),
        "score_malignant": test_scores,
    })

    for _, row in selected_df.iterrows():
        name = row["selection_name"]
        threshold = float(row["selected_threshold_from_val"])
        pred_df[f"pred_{name}"] = (test_scores >= threshold).astype(int)
        pred_df[f"pred_{name}_name"] = pd.Series(pred_df[f"pred_{name}"]).map({0: "benign", 1: "malignant"})

    pred_path = os.path.join(args.output_dir, "test_predictions_thresholded.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    roc_path = os.path.join(args.output_dir, "roi_margin030_threshold_roc.png")
    plot_roc(y_test, test_scores, roc_path, "ROI margin 0.30 threshold tuning ROC")

    summary = {
        "method": "ROI margin 0.30 threshold tuning",
        "seed": int(args.seed),
        "model_path": args.model_path,
        "roi_csv": args.roi_csv,
        "val_sweep_path": val_sweep_path,
        "selected_results_path": selected_path,
        "prediction_path": pred_path,
        "roc_path": roc_path,
        "selected": selected_rows,
    }

    summary_path = os.path.join(args.output_dir, "threshold_tuning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ROI THRESHOLD TUNING COMPLETE")
    print("=" * 80)
    print(f"Validation sweep saved: {val_sweep_path}")
    print(f"Selected test results saved: {selected_path}")
    print(f"Predictions saved: {pred_path}")
    print("\n[Selected threshold test results]")
    print(selected_df[[
        "selection_name",
        "selected_threshold_from_val",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_roc_auc",
        "test_malignant_precision",
        "test_malignant_recall",
        "test_malignant_f1",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold tuning for ROI-aware MMIBC model.")

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
        default="saved_models/best_multimodal_model_roi_ce_margin030_bs4_seed42.pth",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/roi_fusion/roi_margin030_threshold_seed42",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max_threshold",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--macro_tolerance",
        type=float,
        default=0.03,
        help="Allowed validation macro F1 drop for recall_constrained threshold.",
    )

    parser.add_argument(
        "--acc_tolerance",
        type=float,
        default=0.04,
        help="Allowed validation accuracy drop for recall_constrained threshold.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)