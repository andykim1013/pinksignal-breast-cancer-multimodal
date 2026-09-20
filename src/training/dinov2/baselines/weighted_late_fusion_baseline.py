import os
import sys
import json
import argparse
import logging

import yaml
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
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
    """Allow imports such as src.training... when running from Anaconda Prompt."""
    if root not in sys.path:
        sys.path.append(root)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state_dict_robust(model, model_path, device):
    """Load either a plain state_dict or a checkpoint dict."""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    return model


def _get_legacy_roots():
    """
    Optional backward-compatibility hook for replace_old_root().

    If a pairing CSV still stores absolute paths recorded on a different
    (old) machine, set the MMIBC_LEGACY_ROOTS environment variable to a
    ';'-separated list of those old root directories, e.g.:

        MMIBC_LEGACY_ROOTS=<old_root_1>;<old_root_2>

    No old roots are hardcoded in this file. If the environment variable is
    not set (the default for a fresh clone of this repository), this
    returns an empty list and replace_old_root() leaves every path
    untouched.
    """
    raw = os.environ.get("MMIBC_LEGACY_ROOTS", "")
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(";") if entry.strip()]


def replace_old_root(path, root):
    """
    If CSV still contains an old absolute root, replace it with the current root.
    This is safe even when the path is already correct.

    Old roots come only from the MMIBC_LEGACY_ROOTS environment variable
    (see _get_legacy_roots()). When it is unset, this function is a no-op
    and simply returns `path` unchanged.
    """
    if not isinstance(path, str):
        return path

    old_roots = _get_legacy_roots()

    fixed = path
    for old in old_roots:
        if fixed.lower().startswith(old.lower()):
            rel = fixed[len(old):].lstrip("\\/")
            fixed = os.path.join(root, rel)
            break

    return fixed


class PairedUnimodalScoreDataset(Dataset):
    """
    Loads the same paired rows used by MMIBC, but applies modality-specific
    unimodal evaluation transforms:
      - mammography: advanced mammography eval transform
      - ultrasound: medical ultrasound eval transform
    """

    def __init__(self, pair_csv, split, root, image_size):
        from src.training.dinov2.unimodal_model.unimodal_dataset import (
            get_medical_transforms,
            get_advanced_mammo_transforms,
        )

        self.root = root
        self.split = split

        df = pd.read_csv(pair_csv)
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {pair_csv}")

        self.df = df

        _, self.us_transform = get_medical_transforms(image_size)
        _, self.mammo_transform = get_advanced_mammo_transforms(image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mammo_path = replace_old_root(row["mammo_path"], self.root)
        us_path = replace_old_root(row["ultrasound_path"], self.root)

        label = LABEL_MAP[row["label"]]

        mammo_img = Image.open(mammo_path).convert("RGB")
        us_img = Image.open(us_path).convert("RGB")

        mammo_tensor = self.mammo_transform(mammo_img)
        us_tensor = self.us_transform(us_img)

        return {
            "mammo": mammo_tensor,
            "us": us_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "mammo_path": mammo_path,
            "us_path": us_path,
        }


def collect_unimodal_scores(us_model, mammo_model, loader, device):
    """
    Collect malignant probabilities from the already-trained unimodal models.
    score = P(malignant)
    """
    us_model.eval()
    mammo_model.eval()

    labels = []
    us_scores = []
    mammo_scores = []
    mammo_paths = []
    us_paths = []

    with torch.no_grad():
        for batch in loader:
            mammo = batch["mammo"].to(device)
            us = batch["us"].to(device)
            y = batch["label"].cpu().numpy()

            mammo_logits = mammo_model(mammo)
            us_logits = us_model(us)

            mammo_prob = F.softmax(mammo_logits, dim=1)[:, 1].detach().cpu().numpy()
            us_prob = F.softmax(us_logits, dim=1)[:, 1].detach().cpu().numpy()

            labels.extend(y.tolist())
            mammo_scores.extend(mammo_prob.tolist())
            us_scores.extend(us_prob.tolist())
            mammo_paths.extend(batch["mammo_path"])
            us_paths.extend(batch["us_path"])

    return {
        "labels": np.array(labels),
        "mammo_scores": np.array(mammo_scores),
        "us_scores": np.array(us_scores),
        "mammo_paths": mammo_paths,
        "us_paths": us_paths,
    }


def fit_minmax(scores):
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    return min_v, max_v


def apply_minmax(scores, min_v, max_v):
    denom = max_v - min_v
    if denom < 1e-8:
        return np.full_like(scores, 0.5, dtype=np.float32)
    return np.clip((scores - min_v) / denom, 0.0, 1.0)


def youden_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "youden_j": float(j_scores[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1.0 - fpr[best_idx]),
    }


def compute_metrics(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "benign_precision": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "malignant_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }

    return metrics, preds


def sweep_weights(labels, mammo_scores, us_scores):
    """
    Following Tao Tan style:
    weights range 0.01 to 0.99 and sum to 1.
    Here weight_mammo = w, weight_us = 1 - w.
    """
    rows = []

    for w_mammo in np.arange(0.01, 1.00, 0.01):
        w_mammo = round(float(w_mammo), 2)
        w_us = round(1.0 - w_mammo, 2)

        fused = w_mammo * mammo_scores + w_us * us_scores

        auc_value = roc_auc_score(labels, fused)
        th_info = youden_threshold(labels, fused)
        metrics, _ = compute_metrics(labels, fused, th_info["threshold"])

        rows.append({
            "weight_mammo": w_mammo,
            "weight_us": w_us,
            "threshold": th_info["threshold"],
            "youden_j": th_info["youden_j"],
            "val_auc": auc_value,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_malignant_recall": metrics["malignant_recall"],
            "val_specificity": metrics["specificity"],
            "val_sensitivity": metrics["sensitivity"],
        })

    result_df = pd.DataFrame(rows)

    # Tao 논문이 AUC 기준 비교를 했으므로 validation AUC 우선.
    # 동률이면 malignant recall, macro F1 순으로 보조 정렬.
    result_df = result_df.sort_values(
        by=["val_auc", "val_malignant_recall", "val_macro_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return result_df


def plot_roc(labels, scores_dict, save_path):
    plt.figure(figsize=(7, 7))

    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_value = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Weighted Late Fusion ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root = args.root
    setup_project_path(root)

    from src.training.dinov2.unimodal_model.unimodal_model import DinoV2Classifier

    os.makedirs(args.output_dir, exist_ok=True)

    mm_config = load_yaml(args.multimodal_config)
    uni_config = load_yaml(mm_config["models"]["unimodal_config_path"])

    pair_csv = mm_config["data"]["csv_path"]
    us_model_path = mm_config["models"]["us_model_path"]
    mammo_model_path = mm_config["models"]["mammo_model_path"]

    logging.info(f"Project root: {root}")
    logging.info(f"Pair CSV: {pair_csv}")
    logging.info(f"US model: {us_model_path}")
    logging.info(f"Mammo model: {mammo_model_path}")
    logging.info(f"Output dir: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")

    # Load datasets
    image_size = int(mm_config["training"]["image_size"])
    batch_size = int(args.batch_size)

    val_dataset = PairedUnimodalScoreDataset(
        pair_csv=pair_csv,
        split="validation",
        root=root,
        image_size=image_size,
    )
    test_dataset = PairedUnimodalScoreDataset(
        pair_csv=pair_csv,
        split="test",
        root=root,
        image_size=image_size,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load unimodal models
    model_name = uni_config["model"]["name"]
    n_classes = int(uni_config["model"]["n_classes"])
    dropout_rate = float(uni_config["training"]["dropout_rate"])

    us_model = DinoV2Classifier(
        n_classes=n_classes,
        model_name=model_name,
        dropout_rate=dropout_rate,
    ).to(device)

    mammo_model = DinoV2Classifier(
        n_classes=n_classes,
        model_name=model_name,
        dropout_rate=dropout_rate,
    ).to(device)

    us_model = load_state_dict_robust(us_model, us_model_path, device)
    mammo_model = load_state_dict_robust(mammo_model, mammo_model_path, device)

    # Collect scores
    logging.info("Collecting validation scores...")
    val_scores = collect_unimodal_scores(us_model, mammo_model, val_loader, device)

    logging.info("Collecting test scores...")
    test_scores = collect_unimodal_scores(us_model, mammo_model, test_loader, device)

    y_val = val_scores["labels"]
    y_test = test_scores["labels"]

    # Normalize scores to 0~1 following Tao Tan style.
    # Since probabilities are already 0~1, this is mainly for score-level comparability.
    mammo_min, mammo_max = fit_minmax(val_scores["mammo_scores"])
    us_min, us_max = fit_minmax(val_scores["us_scores"])

    val_mammo_norm = apply_minmax(val_scores["mammo_scores"], mammo_min, mammo_max)
    val_us_norm = apply_minmax(val_scores["us_scores"], us_min, us_max)

    test_mammo_norm = apply_minmax(test_scores["mammo_scores"], mammo_min, mammo_max)
    test_us_norm = apply_minmax(test_scores["us_scores"], us_min, us_max)

    # Sweep weights on validation
    logging.info("Sweeping weights on validation split...")
    sweep_df = sweep_weights(y_val, val_mammo_norm, val_us_norm)
    sweep_path = os.path.join(args.output_dir, "validation_weight_search.csv")
    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    best = sweep_df.iloc[0].to_dict()
    best_w_mammo = float(best["weight_mammo"])
    best_w_us = float(best["weight_us"])
    best_threshold = float(best["threshold"])

    # Fixed Tao-style reported weight: DM 0.25 + ABUS 0.75
    fixed_w_mammo = 0.25
    fixed_w_us = 0.75
    fixed_val_fused = fixed_w_mammo * val_mammo_norm + fixed_w_us * val_us_norm
    fixed_threshold = youden_threshold(y_val, fixed_val_fused)["threshold"]

    # Test evaluation
    best_test_fused = best_w_mammo * test_mammo_norm + best_w_us * test_us_norm
    fixed_test_fused = fixed_w_mammo * test_mammo_norm + fixed_w_us * test_us_norm

    best_metrics, best_preds = compute_metrics(y_test, best_test_fused, best_threshold)
    fixed_metrics, fixed_preds = compute_metrics(y_test, fixed_test_fused, fixed_threshold)

    # Save predictions
    pred_df = pd.DataFrame({
        "label": y_test,
        "label_name": ["malignant if x=1 else benign" for _ in y_test],
        "mammo_path": test_scores["mammo_paths"],
        "us_path": test_scores["us_paths"],
        "mammo_score_raw": test_scores["mammo_scores"],
        "us_score_raw": test_scores["us_scores"],
        "mammo_score_norm": test_mammo_norm,
        "us_score_norm": test_us_norm,
        "best_fused_score": best_test_fused,
        "best_pred": best_preds,
        "fixed_025_075_fused_score": fixed_test_fused,
        "fixed_025_075_pred": fixed_preds,
    })

    pred_df["label_name"] = pred_df["label"].map({0: "benign", 1: "malignant"})
    pred_df["best_pred_name"] = pred_df["best_pred"].map({0: "benign", 1: "malignant"})
    pred_df["fixed_025_075_pred_name"] = pred_df["fixed_025_075_pred"].map({0: "benign", 1: "malignant"})

    pred_path = os.path.join(args.output_dir, "late_fusion_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # Save reports
    class_names = ["benign", "malignant"]
    best_report = classification_report(y_test, best_preds, target_names=class_names, zero_division=0)
    fixed_report = classification_report(y_test, fixed_preds, target_names=class_names, zero_division=0)

    report_txt = (
        "=" * 80 + "\n"
        "Weighted Late Fusion Baseline - Tao Tan Style\n"
        "=" * 80 + "\n\n"
        "[Best validation AUC weight]\n"
        f"weight_mammo={best_w_mammo:.2f}, weight_us={best_w_us:.2f}, threshold={best_threshold:.6f}\n"
        f"{json.dumps(best_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{best_report}\n\n"
        "[Fixed Tao reported weight style: mammography 0.25 + ultrasound 0.75]\n"
        f"weight_mammo={fixed_w_mammo:.2f}, weight_us={fixed_w_us:.2f}, threshold={fixed_threshold:.6f}\n"
        f"{json.dumps(fixed_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{fixed_report}\n"
    )

    report_path = os.path.join(args.output_dir, "weighted_late_fusion_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    summary = {
        "method": "Tao Tan style weighted score-level late fusion",
        "normalization": {
            "type": "validation_minmax",
            "mammo_min": mammo_min,
            "mammo_max": mammo_max,
            "us_min": us_min,
            "us_max": us_max,
        },
        "best_validation_auc_weight": {
            "weight_mammo": best_w_mammo,
            "weight_us": best_w_us,
            "threshold": best_threshold,
            "validation_row": best,
            "test_metrics": best_metrics,
        },
        "fixed_025_075_weight": {
            "weight_mammo": fixed_w_mammo,
            "weight_us": fixed_w_us,
            "threshold": fixed_threshold,
            "test_metrics": fixed_metrics,
        },
        "paths": {
            "validation_weight_search": sweep_path,
            "test_predictions": pred_path,
            "report": report_path,
        },
    }

    summary_path = os.path.join(args.output_dir, "weighted_late_fusion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save plots
    roc_path = os.path.join(args.output_dir, "weighted_late_fusion_roc.png")
    plot_roc(
        y_test,
        {
            f"Best sweep wMG={best_w_mammo:.2f}, wUS={best_w_us:.2f}": best_test_fused,
            "Fixed wMG=0.25, wUS=0.75": fixed_test_fused,
        },
        roc_path,
    )

    best_cm_path = os.path.join(args.output_dir, "weighted_late_fusion_confusion_matrix_best.png")
    fixed_cm_path = os.path.join(args.output_dir, "weighted_late_fusion_confusion_matrix_fixed_025_075.png")

    plot_confusion_matrix(
        np.array(best_metrics["confusion_matrix"]),
        best_cm_path,
        f"Weighted Late Fusion CM - Best wMG={best_w_mammo:.2f}, wUS={best_w_us:.2f}",
    )
    plot_confusion_matrix(
        np.array(fixed_metrics["confusion_matrix"]),
        fixed_cm_path,
        "Weighted Late Fusion CM - Fixed wMG=0.25, wUS=0.75",
    )

    logging.info("Done.")
    logging.info(f"Saved validation sweep: {sweep_path}")
    logging.info(f"Saved predictions: {pred_path}")
    logging.info(f"Saved report: {report_path}")
    logging.info(f"Saved summary: {summary_path}")
    logging.info(f"Saved ROC plot: {roc_path}")

    print("\n" + "=" * 80)
    print("BEST VALIDATION AUC WEIGHT - TEST RESULT")
    print("=" * 80)
    print(f"weight_mammo={best_w_mammo:.2f}, weight_us={best_w_us:.2f}, threshold={best_threshold:.6f}")
    print(best_report)

    print("\n" + "=" * 80)
    print("FIXED TAO STYLE WEIGHT 0.25 / 0.75 - TEST RESULT")
    print("=" * 80)
    print(f"weight_mammo=0.25, weight_us=0.75, threshold={fixed_threshold:.6f}")
    print(fixed_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tao Tan style weighted late fusion baseline for MMIBC.")

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root path.",
    )

    parser.add_argument(
        "--multimodal_config",
        type=str,
        default="src/training/dinov2/multimodal_model/config.yaml",
        help="Path to multimodal config.yaml.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baselines/late_fusion_baseline_tao2023",
        help="Directory to save baseline outputs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for score extraction.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    args = parser.parse_args()
    main(args)