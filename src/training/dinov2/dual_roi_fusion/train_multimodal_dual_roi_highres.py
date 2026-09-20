import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from sklearn.metrics import classification_report

from src.training.dinov2.dual_roi_fusion.multimodal_dual_roi_dataset import MultimodalDualROIDataset
from src.training.dinov2.dual_roi_fusion.dual_roi_architecture import DualROIMultimodalFusionModel

from src.training.dinov2.dual_roi_fusion.train_multimodal_dual_roi import (
    setup_project_path,
    set_seed,
    seed_worker,
    load_yaml,
    setup_logging,
    load_state_dict_robust,
    set_train_mode,
    compute_metrics,
    collect_logits,
    run_validation,
    plot_confusion_matrix,
    plot_roc,
)


def main(args):
    setup_project_path(args.root)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    set_seed(args.seed)

    config = load_yaml(args.config)
    unimodal_config = load_yaml(config["models"]["unimodal_config_path"])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    logging.info("Starting 4-3 Dual-ROI high-resolution training")
    logging.info(f"Device: {device}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Image size override: {args.image_size}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Train mode: {args.train_mode}")
    logging.info(f"Init model path: {args.init_model_path}")
    logging.info(f"ROI CSV: {args.roi_csv}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Log file: {log_file}")

    train_dataset = MultimodalDualROIDataset(
        csv_file=args.roi_csv,
        split="train",
        image_size=args.image_size,
        roi_col=args.roi_col,
    )

    val_dataset = MultimodalDualROIDataset(
        csv_file=args.roi_csv,
        split="validation",
        image_size=args.image_size,
        roi_col=args.roi_col,
    )

    test_dataset = MultimodalDualROIDataset(
        csv_file=args.roi_csv,
        split="test",
        image_size=args.image_size,
        roi_col=args.roi_col,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    eval_generator = torch.Generator()
    eval_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

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

    model = DualROIMultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config["models"]["us_model_path"],
        mammo_model_path=config["models"]["mammo_model_path"],
        dropout_rate=float(args.dropout_rate),
    ).to(device)

    if args.init_model_path is not None and args.init_model_path != "" and os.path.exists(args.init_model_path):
        logging.info(f"Loading partial checkpoint from existing MMIBC model: {args.init_model_path}")

        checkpoint = torch.load(args.init_model_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        logging.info(f"Partial load missing keys: {len(missing)}")
        logging.info(f"Partial load unexpected keys: {len(unexpected)}")
    else:
        logging.info("No multimodal checkpoint loaded. Using unimodal encoders + fresh dual fusion layers.")

    trainable_params = set_train_mode(model, args.train_mode)

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check train_mode.")

    logging.info(f"Trainable parameter tensors: {len(trainable_params)}")
    logging.info(f"Model freeze_encoders flag: {model.freeze_encoders}")

    criterion = nn.CrossEntropyLoss()

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

    logging.info("High-resolution training loop started.")
    logging.info(f"Max epochs: {args.epochs}")
    logging.info(f"Patience: {args.patience}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Weight decay: {args.weight_decay}")
    logging.info(f"Model save path: {model_save_path}")

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0

        for mammo_batch, us_batch, us_roi_batch, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [Dual-ROI HighRes Train]",
        ):
            mammo_batch = mammo_batch.to(device)
            us_batch = us_batch.to(device)
            us_roi_batch = us_roi_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(mammo_batch, us_batch, us_roi_batch)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        val_metrics = run_validation(model, val_loader, criterion, device)

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch + 1,
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "train_loss": float(avg_train_loss),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_malignant_precision": val_metrics["malignant_precision"],
            "val_malignant_recall": val_metrics["malignant_recall"],
            "val_malignant_f1": val_metrics["malignant_f1"],
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
            logging.info(f"Early stopping counter: {patience_counter} / {args.patience}")

        if patience_counter >= int(args.patience):
            logging.info("Early stopping triggered.")
            break

    writer.close()

    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, "dual_roi_highres_training_history.csv")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    logging.info("Loading best high-resolution Dual-ROI model for final test evaluation.")
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
        "4-3 Dual-ROI High-Resolution MMIBC Clean Test Result\n"
        "=" * 80 + "\n\n"
        f"Seed: {args.seed}\n"
        f"Best epoch: {best_epoch}\n"
        f"Image size: {args.image_size}\n"
        f"Batch size: {args.batch_size}\n"
        f"Train mode: {args.train_mode}\n"
        f"Init model path: {args.init_model_path}\n"
        f"ROI CSV: {args.roi_csv}\n"
        f"Model path: {model_save_path}\n\n"
        "[Clean Test Metrics]\n"
        f"{json.dumps(test_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"{test_report}\n"
    )

    report_path = os.path.join(args.output_dir, "dual_roi_highres_clean_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    summary = {
        "method": "4-3 Dual-ROI High-Resolution MMIBC",
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "config": args.config,
        "roi_csv": args.roi_csv,
        "roi_col": args.roi_col,
        "train_mode": args.train_mode,
        "init_model_path": args.init_model_path,
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

    summary_path = os.path.join(args.output_dir, "dual_roi_highres_clean_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_df = pd.DataFrame({
        "label": y_test,
        "label_name": pd.Series(y_test).map({0: "benign", 1: "malignant"}),
        "score_malignant": test_scores,
        "pred": test_preds,
        "pred_name": pd.Series(test_preds).map({0: "benign", 1: "malignant"}),
    })

    pred_path = os.path.join(args.output_dir, "dual_roi_highres_clean_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        os.path.join(args.output_dir, "cm_dual_roi_highres_clean.png"),
        "4-3 Dual-ROI HighRes Clean Test",
    )

    plot_roc(
        y_test,
        test_scores,
        os.path.join(args.output_dir, "roc_dual_roi_highres_clean.png"),
        "4-3 Dual-ROI HighRes ROC - Clean Test",
    )

    logging.info("High-resolution Dual-ROI training finished.")
    logging.info(f"History saved: {history_path}")
    logging.info(f"Report saved: {report_path}")
    logging.info(f"Summary saved: {summary_path}")
    logging.info(f"Predictions saved: {pred_path}")

    print("\n" + "=" * 80)
    print("4-3 DUAL-ROI HIGH-RESOLUTION MMIBC CLEAN TEST RESULT")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train mode: {args.train_mode}")
    print(f"Model path: {model_save_path}")
    print(test_report)

    print("\n" + "=" * 80)
    print("ADDITIONAL TEST METRICS")
    print("=" * 80)
    print(f"Accuracy           : {test_metrics['accuracy']:.4f}")
    print(f"Macro F1           : {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1        : {test_metrics['weighted_f1']:.4f}")
    print(f"ROC AUC            : {test_metrics['roc_auc']:.4f}")
    print(f"Benign Precision   : {test_metrics['benign_precision']:.4f}")
    print(f"Benign Recall      : {test_metrics['benign_recall']:.4f}")
    print(f"Malignant Precision: {test_metrics['malignant_precision']:.4f}")
    print(f"Malignant Recall   : {test_metrics['malignant_recall']:.4f}")
    print(f"Malignant F1       : {test_metrics['malignant_f1']:.4f}")
    print(f"TN / FP / FN / TP  : {test_metrics['tn']} / {test_metrics['fp']} / {test_metrics['fn']} / {test_metrics['tp']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-3 Dual-ROI high-resolution MMIBC training.")

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
        "--init_model_path",
        type=str,
        default="saved_models/best_multimodal_model.pth",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dual_roi_fusion/dual_roi_highres384_bs2_seed42_init_mmibc",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="saved_models",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="best_multimodal_model_dual_roi_highres384_bs2_seed42_init_mmibc.pth",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        default="fusion_layers",
        choices=["context_only", "classifier_only", "fusion_layers", "all"],
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
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
        "--dropout_rate",
        type=float,
        default=0.3,
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
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)