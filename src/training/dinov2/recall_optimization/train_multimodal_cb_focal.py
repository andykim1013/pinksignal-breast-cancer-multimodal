# import os
# import sys
# import json
# import argparse
# import logging
# from datetime import datetime

# import yaml
# import numpy as np
# import pandas as pd

# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F

# from tqdm import tqdm
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
#     roc_curve,
# )

# import matplotlib.pyplot as plt
# import seaborn as sns


# LABEL_MAP = {
#     "benign": 0,
#     "malignant": 1,
# }


# def setup_project_path(root):
#     if root not in sys.path:
#         sys.path.append(root)


# def load_yaml(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def setup_logging(output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_dir = os.path.join(output_dir, f"logs_cb_focal_{current_time}")
#     os.makedirs(log_dir, exist_ok=True)

#     log_file = os.path.join(log_dir, "training_run.log")

#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, encoding="utf-8"),
#             logging.StreamHandler(),
#         ],
#     )

#     return log_file


# def load_state_dict_robust(model, model_path, device, strict=True):
#     checkpoint = torch.load(model_path, map_location=device)

#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         checkpoint = checkpoint["model_state_dict"]
#     elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         checkpoint = checkpoint["state_dict"]

#     model.load_state_dict(checkpoint, strict=strict)
#     return model


# def get_train_class_counts(csv_path):
#     df = pd.read_csv(csv_path)
#     train_df = df[df["split"] == "train"].copy()

#     if len(train_df) == 0:
#         raise ValueError("No train rows found in CSV.")

#     labels = train_df["label"].map(LABEL_MAP).values
#     counts = np.bincount(labels, minlength=2)

#     return counts.tolist()


# def compute_metrics(labels, logits):
#     probs = F.softmax(torch.tensor(logits), dim=1).numpy()
#     scores = probs[:, 1]
#     preds = np.argmax(logits, axis=1)

#     cm = confusion_matrix(labels, preds, labels=[0, 1])
#     tn, fp, fn, tp = cm.ravel()

#     metrics = {
#         "accuracy": float(accuracy_score(labels, preds)),
#         "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
#         "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
#         "roc_auc": float(roc_auc_score(labels, scores)),
#         "benign_precision": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
#         "benign_recall": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
#         "malignant_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
#         "malignant_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
#         "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
#         "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
#         "confusion_matrix": cm.tolist(),
#     }

#     return metrics, preds, scores


# def collect_logits(model, loader, device):
#     model.eval()

#     all_labels = []
#     all_logits = []

#     with torch.no_grad():
#         for mammo_batch, us_batch, labels in loader:
#             mammo_batch = mammo_batch.to(device)
#             us_batch = us_batch.to(device)
#             labels = labels.to(device)

#             outputs = model(mammo_batch, us_batch)

#             all_logits.append(outputs.detach().cpu())
#             all_labels.extend(labels.detach().cpu().numpy().tolist())

#     all_logits = torch.cat(all_logits, dim=0).numpy()
#     all_labels = np.array(all_labels)

#     return all_labels, all_logits


# def run_validation(model, loader, criterion, device):
#     model.eval()

#     total_loss = 0.0
#     all_labels = []
#     all_logits = []

#     with torch.no_grad():
#         for mammo_batch, us_batch, labels in loader:
#             mammo_batch = mammo_batch.to(device)
#             us_batch = us_batch.to(device)
#             labels = labels.to(device)

#             outputs = model(mammo_batch, us_batch)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             all_logits.append(outputs.detach().cpu())
#             all_labels.extend(labels.detach().cpu().numpy().tolist())

#     avg_loss = total_loss / max(len(loader), 1)

#     all_logits = torch.cat(all_logits, dim=0).numpy()
#     all_labels = np.array(all_labels)

#     metrics, _, _ = compute_metrics(all_labels, all_logits)
#     metrics["loss"] = float(avg_loss)

#     return metrics


# def plot_confusion_matrix(cm, save_path, title):
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="g",
#         cmap="Blues",
#         xticklabels=["benign", "malignant"],
#         yticklabels=["benign", "malignant"],
#     )
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


# def plot_roc(labels, scores, save_path, title):
#     fpr, tpr, _ = roc_curve(labels, scores)
#     auc_value = roc_auc_score(labels, scores)

#     plt.figure(figsize=(7, 7))
#     plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_value:.3f}")
#     plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
#     plt.xlabel("1 - Specificity")
#     plt.ylabel("Sensitivity")
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


# def main(args):
#     setup_project_path(args.root)

#     from src.training.dinov2.multimodal_model.multimodal_dataset import MultimodalDataset
#     from src.training.dinov2.multimodal_model.multimodal_architecture import MultimodalFusionModel
#     from src.training.dinov2.recall_optimization.losses import ClassBalancedFocalLoss

#     os.makedirs(args.output_dir, exist_ok=True)
#     log_file = setup_logging(args.output_dir)

#     config = load_yaml(args.config)
#     unimodal_config = load_yaml(config["models"]["unimodal_config_path"])

#     device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

#     samples_per_class = get_train_class_counts(config["data"]["csv_path"])

#     logging.info("Starting 2-3 Class-Balanced Focal Loss training")
#     logging.info(f"Using device: {device}")
#     logging.info(f"Config: {args.config}")
#     logging.info(f"CSV: {config['data']['csv_path']}")
#     logging.info(f"Train class counts [benign, malignant]: {samples_per_class}")
#     logging.info(f"Beta: {args.beta}")
#     logging.info(f"Gamma: {args.gamma}")
#     logging.info(f"Init model path: {args.init_model_path}")
#     logging.info(f"Output dir: {args.output_dir}")
#     logging.info(f"Log file: {log_file}")

#     train_dataset = MultimodalDataset(
#         csv_file=config["data"]["csv_path"],
#         split="train",
#         image_size=config["training"]["image_size"],
#     )

#     val_dataset = MultimodalDataset(
#         csv_file=config["data"]["csv_path"],
#         split="validation",
#         image_size=config["training"]["image_size"],
#     )

#     test_dataset = MultimodalDataset(
#         csv_file=config["data"]["csv_path"],
#         split="test",
#         image_size=config["training"]["image_size"],
#     )

#     batch_size = int(args.batch_size)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#     )

#     model = MultimodalFusionModel(
#         unimodal_config=unimodal_config,
#         us_model_path=config["models"]["us_model_path"],
#         mammo_model_path=config["models"]["mammo_model_path"],
#     ).to(device)

#     if args.init_model_path is not None and os.path.exists(args.init_model_path):
#         logging.info("Loading existing clean MMIBC multimodal model before CB-Focal fine-tuning.")
#         model = load_state_dict_robust(model, args.init_model_path, device, strict=True)
#     else:
#         logging.warning("No init model loaded. Training from freshly initialized multimodal fusion layers.")

#     criterion = ClassBalancedFocalLoss(
#         samples_per_class=samples_per_class,
#         beta=float(args.beta),
#         gamma=float(args.gamma),
#         reduction="mean",
#     )

#     logging.info(f"CB-Focal class weights: {criterion.weights.detach().cpu().numpy().tolist()}")

#     trainable_params = [p for p in model.parameters() if p.requires_grad]

#     logging.info(f"Trainable parameter tensors: {len(trainable_params)}")

#     optimizer = optim.AdamW(
#         trainable_params,
#         lr=float(args.learning_rate),
#         weight_decay=float(args.weight_decay),
#     )

#     scheduler = ReduceLROnPlateau(
#         optimizer,
#         mode="max",
#         patience=5,
#         factor=0.5,
#     )

#     tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
#     writer = SummaryWriter(log_dir=tensorboard_dir)

#     os.makedirs(args.model_save_dir, exist_ok=True)
#     model_save_path = os.path.join(args.model_save_dir, args.model_name)

#     history = []

#     best_macro_f1 = -1.0
#     best_malignant_recall = -1.0
#     best_epoch = -1
#     patience_counter = 0

#     max_epochs = int(args.epochs)
#     patience = int(args.patience)

#     logging.info("Training loop started.")
#     logging.info(f"Max epochs: {max_epochs}")
#     logging.info(f"Early stopping patience: {patience}")
#     logging.info(f"Batch size: {batch_size}")
#     logging.info(f"Learning rate: {args.learning_rate}")
#     logging.info(f"Model save path: {model_save_path}")

#     for epoch in range(max_epochs):
#         model.train()
#         train_loss = 0.0

#         for mammo_batch, us_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [CB-Focal Train]"):
#             mammo_batch = mammo_batch.to(device)
#             us_batch = us_batch.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(mammo_batch, us_batch)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#         avg_train_loss = train_loss / max(len(train_loader), 1)
#         val_metrics = run_validation(model, val_loader, criterion, device)

#         scheduler.step(val_metrics["macro_f1"])

#         row = {
#             "epoch": epoch + 1,
#             "train_loss": float(avg_train_loss),
#             "val_loss": val_metrics["loss"],
#             "val_accuracy": val_metrics["accuracy"],
#             "val_macro_f1": val_metrics["macro_f1"],
#             "val_weighted_f1": val_metrics["weighted_f1"],
#             "val_roc_auc": val_metrics["roc_auc"],
#             "val_malignant_precision": val_metrics["malignant_precision"],
#             "val_malignant_recall": val_metrics["malignant_recall"],
#             "val_specificity": val_metrics["specificity"],
#             "val_sensitivity": val_metrics["sensitivity"],
#         }

#         history.append(row)

#         writer.add_scalar("Loss/train", avg_train_loss, epoch)
#         writer.add_scalar("Loss/validation", val_metrics["loss"], epoch)
#         writer.add_scalar("Metric/val_accuracy", val_metrics["accuracy"], epoch)
#         writer.add_scalar("Metric/val_macro_f1", val_metrics["macro_f1"], epoch)
#         writer.add_scalar("Metric/val_malignant_recall", val_metrics["malignant_recall"], epoch)
#         writer.add_scalar("Metric/val_roc_auc", val_metrics["roc_auc"], epoch)

#         logging.info(
#             f"Epoch {epoch + 1}: "
#             f"Train Loss={avg_train_loss:.4f}, "
#             f"Val Loss={val_metrics['loss']:.4f}, "
#             f"Val Acc={val_metrics['accuracy']:.4f}, "
#             f"Val MacroF1={val_metrics['macro_f1']:.4f}, "
#             f"Val MalRecall={val_metrics['malignant_recall']:.4f}, "
#             f"Val AUC={val_metrics['roc_auc']:.4f}"
#         )

#         improved = False

#         if val_metrics["macro_f1"] > best_macro_f1:
#             improved = True
#         elif val_metrics["macro_f1"] == best_macro_f1 and val_metrics["malignant_recall"] > best_malignant_recall:
#             improved = True

#         if improved:
#             best_macro_f1 = val_metrics["macro_f1"]
#             best_malignant_recall = val_metrics["malignant_recall"]
#             best_epoch = epoch + 1
#             patience_counter = 0

#             torch.save(model.state_dict(), model_save_path)

#             logging.info(
#                 f"Best model updated at epoch {best_epoch}: "
#                 f"MacroF1={best_macro_f1:.4f}, "
#                 f"MalRecall={best_malignant_recall:.4f}"
#             )
#         else:
#             patience_counter += 1
#             logging.info(f"Early stopping counter: {patience_counter} / {patience}")

#         if patience_counter >= patience:
#             logging.info("Early stopping triggered.")
#             break

#     writer.close()

#     history_df = pd.DataFrame(history)
#     history_path = os.path.join(args.output_dir, "cb_focal_training_history.csv")
#     history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

#     logging.info("Loading best CB-Focal model for final clean test evaluation.")
#     model = load_state_dict_robust(model, model_save_path, device)
#     model.eval()

#     y_test, test_logits = collect_logits(model, test_loader, device)
#     test_metrics, test_preds, test_scores = compute_metrics(y_test, test_logits)

#     test_report = classification_report(
#         y_test,
#         test_preds,
#         target_names=["benign", "malignant"],
#         zero_division=0,
#     )

#     report_txt = (
#         "=" * 80 + "\n"
#         "2-3 Class-Balanced Focal Loss Clean Test Result\n"
#         "=" * 80 + "\n\n"
#         f"Best epoch: {best_epoch}\n"
#         f"Beta: {args.beta}\n"
#         f"Gamma: {args.gamma}\n"
#         f"Samples per class [benign, malignant]: {samples_per_class}\n"
#         f"Model path: {model_save_path}\n\n"
#         "[Clean Test Metrics]\n"
#         f"{json.dumps(test_metrics, indent=2, ensure_ascii=False)}\n\n"
#         f"{test_report}\n"
#     )

#     report_path = os.path.join(args.output_dir, "cb_focal_clean_test_report.txt")
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(report_txt)

#     summary = {
#         "method": "2-3 Class-Balanced Focal Loss",
#         "config": args.config,
#         "csv": config["data"]["csv_path"],
#         "init_model_path": args.init_model_path,
#         "beta": float(args.beta),
#         "gamma": float(args.gamma),
#         "samples_per_class": {
#             "benign": int(samples_per_class[0]),
#             "malignant": int(samples_per_class[1]),
#         },
#         "class_balanced_weights": criterion.weights.detach().cpu().numpy().tolist(),
#         "best_epoch": int(best_epoch),
#         "best_val_macro_f1": float(best_macro_f1),
#         "best_val_malignant_recall": float(best_malignant_recall),
#         "model_save_path": model_save_path,
#         "test_metrics": test_metrics,
#         "paths": {
#             "history": history_path,
#             "report": report_path,
#             "tensorboard": tensorboard_dir,
#         },
#     }

#     summary_path = os.path.join(args.output_dir, "cb_focal_clean_summary.json")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)

#     pred_df = pd.DataFrame({
#         "label": y_test,
#         "label_name": pd.Series(y_test).map({0: "benign", 1: "malignant"}),
#         "score_malignant": test_scores,
#         "pred": test_preds,
#         "pred_name": pd.Series(test_preds).map({0: "benign", 1: "malignant"}),
#     })

#     pred_path = os.path.join(args.output_dir, "cb_focal_clean_test_predictions.csv")
#     pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

#     plot_confusion_matrix(
#         np.array(test_metrics["confusion_matrix"]),
#         os.path.join(args.output_dir, "cm_cb_focal_clean.png"),
#         "2-3 CB-Focal Clean Test",
#     )

#     plot_roc(
#         y_test,
#         test_scores,
#         os.path.join(args.output_dir, "roc_cb_focal_clean.png"),
#         "2-3 CB-Focal ROC - Clean Test",
#     )

#     logging.info("CB-Focal training finished.")
#     logging.info(f"History saved: {history_path}")
#     logging.info(f"Report saved: {report_path}")
#     logging.info(f"Summary saved: {summary_path}")
#     logging.info(f"Predictions saved: {pred_path}")

#     print("\n" + "=" * 80)
#     print("2-3 CLASS-BALANCED FOCAL LOSS CLEAN TEST RESULT")
#     print("=" * 80)
#     print(f"Best epoch: {best_epoch}")
#     print(f"Model path: {model_save_path}")
#     print(test_report)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="2-3 Class-Balanced Focal Loss training for MMIBC.")

#     parser.add_argument(
#         "--root",
#         type=str,
#         default=".",
#     )

#     parser.add_argument(
#         "--config",
#         type=str,
#         default="src/training/dinov2/multimodal_model/config_no_mammo_leak.yaml",
#     )

#     parser.add_argument(
#         "--init_model_path",
#         type=str,
#         default="saved_models/best_multimodal_model.pth",
#         help="Existing clean MMIBC multimodal model used as initialization.",
#     )

#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="outputs/recall_optimization/cb_focal_clean_bs2",
#     )

#     parser.add_argument(
#         "--model_save_dir",
#         type=str,
#         default="saved_models",
#     )

#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="best_multimodal_model_cb_focal_clean_bs2.pth",
#     )

#     parser.add_argument(
#         "--beta",
#         type=float,
#         default=0.999,
#         help="Class-balanced beta.",
#     )

#     parser.add_argument(
#         "--gamma",
#         type=float,
#         default=0.5,
#         help="Focal gamma. Cui et al. commonly report beta=0.999 and gamma=0.5 for CB-Focal.",
#     )

#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=2,
#     )

#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=50,
#     )

#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=1e-4,
#     )

#     parser.add_argument(
#         "--weight_decay",
#         type=float,
#         default=0.05,
#     )

#     parser.add_argument(
#         "--patience",
#         type=int,
#         default=15,
#     )

#     parser.add_argument(
#         "--cpu",
#         action="store_true",
#     )

#     args = parser.parse_args()
#     main(args)
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm
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
import seaborn as sns


LABEL_MAP = {
    "benign": 0,
    "malignant": 1,
}


def setup_project_path(root):
    if root not in sys.path:
        sys.path.append(root)


def set_seed(seed=42):
    """
    Fix random seeds for reproducible multimodal training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

    logging.info(f"Global training seed fixed: {seed}")


def seed_worker(worker_id):
    """
    Make DataLoader worker randomness reproducible.
    Mainly useful when num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, f"logs_cb_focal_{current_time}")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "training_run.log")

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


def load_state_dict_robust(model, model_path, device, strict=True):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=strict)
    return model


def get_train_class_counts(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].copy()

    if len(train_df) == 0:
        raise ValueError("No train rows found in CSV.")

    labels = train_df["label"].map(LABEL_MAP).values
    counts = np.bincount(labels, minlength=2)

    return counts.tolist()


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
        "malignant_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "malignant_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }

    return metrics, preds, scores


def collect_logits(model, loader, device):
    model.eval()

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for mammo_batch, us_batch, labels in loader:
            mammo_batch = mammo_batch.to(device)
            us_batch = us_batch.to(device)
            labels = labels.to(device)

            outputs = model(mammo_batch, us_batch)

            all_logits.append(outputs.detach().cpu())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = np.array(all_labels)

    return all_labels, all_logits


def run_validation(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for mammo_batch, us_batch, labels in loader:
            mammo_batch = mammo_batch.to(device)
            us_batch = us_batch.to(device)
            labels = labels.to(device)

            outputs = model(mammo_batch, us_batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_logits.append(outputs.detach().cpu())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = np.array(all_labels)

    metrics, _, _ = compute_metrics(all_labels, all_logits)
    metrics["loss"] = float(avg_loss)

    return metrics


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


def main(args):
    setup_project_path(args.root)

    from src.training.dinov2.multimodal_model.multimodal_dataset import MultimodalDataset
    from src.training.dinov2.multimodal_model.multimodal_architecture import MultimodalFusionModel
    from src.training.dinov2.recall_optimization.losses import ClassBalancedFocalLoss

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    set_seed(args.seed)

    config = load_yaml(args.config)
    unimodal_config = load_yaml(config["models"]["unimodal_config_path"])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    samples_per_class = get_train_class_counts(config["data"]["csv_path"])

    logging.info("Starting 2-3 Class-Balanced Focal Loss training")
    logging.info(f"Using device: {device}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Config: {args.config}")
    logging.info(f"CSV: {config['data']['csv_path']}")
    logging.info(f"Train class counts [benign, malignant]: {samples_per_class}")
    logging.info(f"Beta: {args.beta}")
    logging.info(f"Gamma: {args.gamma}")
    logging.info(f"Init model path: {args.init_model_path}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Log file: {log_file}")

    train_dataset = MultimodalDataset(
        csv_file=config["data"]["csv_path"],
        split="train",
        image_size=config["training"]["image_size"],
    )

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

    batch_size = int(args.batch_size)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    eval_generator = torch.Generator()
    eval_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
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

    if args.init_model_path is not None and os.path.exists(args.init_model_path):
        logging.info("Loading existing clean MMIBC multimodal model before CB-Focal fine-tuning.")
        model = load_state_dict_robust(model, args.init_model_path, device, strict=True)
    else:
        logging.warning("No init model loaded. Training from freshly initialized multimodal fusion layers.")

    criterion = ClassBalancedFocalLoss(
        samples_per_class=samples_per_class,
        beta=float(args.beta),
        gamma=float(args.gamma),
        reduction="mean",
    )

    logging.info(f"CB-Focal class weights: {criterion.weights.detach().cpu().numpy().tolist()}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Trainable parameter tensors: {len(trainable_params)}")

    optimizer = optim.AdamW(
        trainable_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
    )

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)

    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_save_dir, args.model_name)

    history = []

    best_macro_f1 = -1.0
    best_malignant_recall = -1.0
    best_epoch = -1
    patience_counter = 0

    max_epochs = int(args.epochs)
    patience = int(args.patience)

    logging.info("Training loop started.")
    logging.info(f"Max epochs: {max_epochs}")
    logging.info(f"Early stopping patience: {patience}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Model save path: {model_save_path}")

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for mammo_batch, us_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [CB-Focal Train]"):
            mammo_batch = mammo_batch.to(device)
            us_batch = us_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(mammo_batch, us_batch)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        val_metrics = run_validation(model, val_loader, criterion, device)

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_malignant_precision": val_metrics["malignant_precision"],
            "val_malignant_recall": val_metrics["malignant_recall"],
            "val_specificity": val_metrics["specificity"],
            "val_sensitivity": val_metrics["sensitivity"],
        }

        history.append(row)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", val_metrics["loss"], epoch)
        writer.add_scalar("Metric/val_accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("Metric/val_macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("Metric/val_malignant_recall", val_metrics["malignant_recall"], epoch)
        writer.add_scalar("Metric/val_roc_auc", val_metrics["roc_auc"], epoch)

        logging.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}, "
            f"Val MacroF1={val_metrics['macro_f1']:.4f}, "
            f"Val MalRecall={val_metrics['malignant_recall']:.4f}, "
            f"Val AUC={val_metrics['roc_auc']:.4f}"
        )

        improved = False

        if val_metrics["macro_f1"] > best_macro_f1:
            improved = True
        elif val_metrics["macro_f1"] == best_macro_f1 and val_metrics["malignant_recall"] > best_malignant_recall:
            improved = True

        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_malignant_recall = val_metrics["malignant_recall"]
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(model.state_dict(), model_save_path)

            logging.info(
                f"Best model updated at epoch {best_epoch}: "
                f"MacroF1={best_macro_f1:.4f}, "
                f"MalRecall={best_malignant_recall:.4f}"
            )
        else:
            patience_counter += 1
            logging.info(f"Early stopping counter: {patience_counter} / {patience}")

        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    writer.close()

    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, "cb_focal_training_history.csv")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    logging.info("Loading best CB-Focal model for final clean test evaluation.")
    model = load_state_dict_robust(model, model_save_path, device)
    model.eval()

    y_test, test_logits = collect_logits(model, test_loader, device)
    test_metrics, test_preds, test_scores = compute_metrics(y_test, test_logits)

    test_report = classification_report(
        y_test,
        test_preds,
        target_names=["benign", "malignant"],
        zero_division=0,
    )

    report_txt = (
        "=" * 80 + "\n"
        "2-3 Class-Balanced Focal Loss Clean Test Result\n"
        "=" * 80 + "\n\n"
        f"Seed: {args.seed}\n"
        f"Best epoch: {best_epoch}\n"
        f"Beta: {args.beta}\n"
        f"Gamma: {args.gamma}\n"
        f"Samples per class [benign, malignant]: {samples_per_class}\n"
        f"Model path: {model_save_path}\n\n"
        "[Clean Test Metrics]\n"
        f"{json.dumps(test_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{test_report}\n"
    )

    report_path = os.path.join(args.output_dir, "cb_focal_clean_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    summary = {
        "method": "2-3 Class-Balanced Focal Loss",
        "seed": int(args.seed),
        "config": args.config,
        "csv": config["data"]["csv_path"],
        "init_model_path": args.init_model_path,
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "samples_per_class": {
            "benign": int(samples_per_class[0]),
            "malignant": int(samples_per_class[1]),
        },
        "class_balanced_weights": criterion.weights.detach().cpu().numpy().tolist(),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_macro_f1),
        "best_val_malignant_recall": float(best_malignant_recall),
        "model_save_path": model_save_path,
        "test_metrics": test_metrics,
        "paths": {
            "history": history_path,
            "report": report_path,
            "tensorboard": tensorboard_dir,
        },
    }

    summary_path = os.path.join(args.output_dir, "cb_focal_clean_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_df = pd.DataFrame({
        "label": y_test,
        "label_name": pd.Series(y_test).map({0: "benign", 1: "malignant"}),
        "score_malignant": test_scores,
        "pred": test_preds,
        "pred_name": pd.Series(test_preds).map({0: "benign", 1: "malignant"}),
    })

    pred_path = os.path.join(args.output_dir, "cb_focal_clean_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        os.path.join(args.output_dir, "cm_cb_focal_clean.png"),
        "2-3 CB-Focal Clean Test",
    )

    plot_roc(
        y_test,
        test_scores,
        os.path.join(args.output_dir, "roc_cb_focal_clean.png"),
        "2-3 CB-Focal ROC - Clean Test",
    )

    logging.info("CB-Focal training finished.")
    logging.info(f"History saved: {history_path}")
    logging.info(f"Report saved: {report_path}")
    logging.info(f"Summary saved: {summary_path}")
    logging.info(f"Predictions saved: {pred_path}")

    print("\n" + "=" * 80)
    print("2-3 CLASS-BALANCED FOCAL LOSS CLEAN TEST RESULT")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Model path: {model_save_path}")
    print(test_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-3 Class-Balanced Focal Loss training for MMIBC.")

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
        "--init_model_path",
        type=str,
        default="saved_models/best_multimodal_model.pth",
        help="Existing clean MMIBC multimodal model used as initialization.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/recall_optimization/cb_focal_clean_bs2_seed42",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="saved_models",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="best_multimodal_model_cb_focal_clean_bs2_seed42.pth",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.999,
        help="Class-balanced beta.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Focal gamma for CB-Focal.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)