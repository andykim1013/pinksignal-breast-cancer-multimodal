# src/training/dinov2/dual_roi_fusion/train_multimodal_dual_roi_classweight3p0_launcher.py
#
# Purpose:
#   Apply class-weighted CrossEntropyLoss to the existing Dual-ROI MMIBC training script
#   without modifying the original Dual-ROI architecture, dataset, or training code.
#
# Class weight:
#   class 0: benign     -> 1.0
#   class 1: malignant -> 3.0
#
# Why this launcher exists:
#   The uploaded class-weight code was originally for the basic MMIBC multimodal model.
#   This launcher transfers only the necessary part:
#       nn.CrossEntropyLoss(weight=[1.0, 3.0])
#   into the current Dual-ROI training pipeline.
#
# Usage example:
#   python src\training\dinov2\dual_roi_fusion\train_multimodal_dual_roi_classweight3p0_launcher.py ^
#     --source_script "src\training\dinov2\dual_roi_fusion\train_multimodal_dual_roi_ce.py" ^
#     --image_size 224 ^
#     --batch_size 4 ^
#     --seed 42 ^
#     --train_mode fusion_layers ^
#     --init_model_path "saved_models\best_multimodal_model.pth" ^
#     --roi_csv "data\multimodal_pairs_roi_margin030_no_mammo_leak.csv" ^
#     --output_dir "outputs\dual_roi_fusion\dual_roi_classweight3p0_bs4_seed42_init_mmibc" ^
#     --model_name best_multimodal_model_dual_roi_classweight3p0_bs4_seed42_init_mmibc.pth

import os
import sys
import runpy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Project root setting
# ============================================================

THIS_FILE = Path(__file__).resolve()

# This file location:
# MMIBC-main/src/training/dinov2/dual_roi_fusion/train_multimodal_dual_roi_classweight3p0_launcher.py
# parents[4] = MMIBC-main
PROJECT_ROOT = THIS_FILE.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Weighted CrossEntropyLoss
# ============================================================

_ORIGINAL_TORCH_NN_CE = nn.CrossEntropyLoss


class DynamicClassWeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss with fixed class weights:
      class 0: benign     -> 1.0
      class 1: malignant -> 3.0

    The weight tensor is moved automatically to the same device as input logits.
    This prevents device mismatch errors such as:
      Expected all tensors to be on the same device
    """

    def __init__(
        self,
        class_weights=(1.0, 3.0),
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ):
        super().__init__()

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Keep these for compatibility with old PyTorch signatures.
        self.size_average = size_average
        self.reduce = reduce

        print("=" * 80)
        print("CLASS WEIGHT MODE ENABLED")
        print("Using weighted CrossEntropyLoss")
        print(f"class 0 benign     weight = {float(class_weights[0])}")
        print(f"class 1 malignant weight = {float(class_weights[1])}")
        print("=" * 80)

    def forward(self, input, target):
        weight = self.class_weights.to(device=input.device, dtype=input.dtype)

        return F.cross_entropy(
            input=input,
            target=target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


def patched_cross_entropy_loss(*args, **kwargs):
    """
    Replacement for nn.CrossEntropyLoss.

    If the original training script calls:
        nn.CrossEntropyLoss()

    this function returns:
        DynamicClassWeightedCrossEntropyLoss([1.0, 3.0])

    If the original code already passes a weight manually, we do not override it.
    This avoids double-applying weights.
    """

    if "weight" in kwargs and kwargs["weight"] is not None:
        print("=" * 80)
        print("WARNING: Source script already passed weight to CrossEntropyLoss.")
        print("Using the source script's own weight instead of classweight3p0 launcher.")
        print("=" * 80)
        return _ORIGINAL_TORCH_NN_CE(*args, **kwargs)

    if len(args) >= 1 and args[0] is not None:
        print("=" * 80)
        print("WARNING: Source script passed positional weight to CrossEntropyLoss.")
        print("Using the source script's own weight instead of classweight3p0 launcher.")
        print("=" * 80)
        return _ORIGINAL_TORCH_NN_CE(*args, **kwargs)

    # Preserve commonly used CrossEntropyLoss options if the source script used them.
    supported_keys = {
        "size_average",
        "ignore_index",
        "reduce",
        "reduction",
        "label_smoothing",
    }

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_keys
    }

    return DynamicClassWeightedCrossEntropyLoss(
        class_weights=(1.0, 3.0),
        **filtered_kwargs,
    )


# ============================================================
# 3. Launcher
# ============================================================

def resolve_path(path_text):
    path = Path(path_text)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run an existing Dual-ROI MMIBC training script with "
            "class-weighted CrossEntropyLoss [1.0, 3.0]."
        ),
        add_help=True,
    )

    parser.add_argument(
        "--source_script",
        type=str,
        required=True,
        help=(
            "Path to the existing successful Dual-ROI training script. "
            "Use the ROI model training script before high-resolution modification."
        ),
    )

    # Parse only launcher-specific argument.
    # The remaining args are passed directly to the source training script.
    launcher_args, source_args = parser.parse_known_args()

    source_script_path = resolve_path(launcher_args.source_script)

    if not source_script_path.exists():
        raise FileNotFoundError(
            f"Source training script not found: {source_script_path}"
        )

    print("=" * 80)
    print("DUAL-ROI CLASSWEIGHT 3.0 LAUNCHER")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source script: {source_script_path}")
    print("Class weights: benign=1.0, malignant=3.0")
    print("The original source script will be executed without modifying the file.")
    print("=" * 80)

    # Patch nn.CrossEntropyLoss globally before running the source script.
    nn.CrossEntropyLoss = patched_cross_entropy_loss
    torch.nn.CrossEntropyLoss = patched_cross_entropy_loss

    # Make the source script think it was directly executed.
    sys.argv = [str(source_script_path)] + source_args

    print("Forwarded arguments to source script:")
    for arg in sys.argv:
        print(f"  {arg}")
    print("=" * 80)

    # Execute original training script as __main__.
    runpy.run_path(str(source_script_path), run_name="__main__")


if __name__ == "__main__":
    main()