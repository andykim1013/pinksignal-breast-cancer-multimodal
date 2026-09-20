# src/training/dinov2/dual_roi_fusion/train_multimodal_dual_roi_classweight_tunable_launcher.py
#
# Purpose:
#   Run the existing Dual-ROI MMIBC training script with tunable class-weighted
#   CrossEntropyLoss, without modifying the original Dual-ROI architecture,
#   dataset, or train_multimodal_dual_roi.py.
#
# Main idea:
#   Original ROI model:
#       criterion = nn.CrossEntropyLoss()
#
#   This launcher changes it at runtime to:
#       criterion = nn.CrossEntropyLoss(weight=[benign_weight, malignant_weight])
#
# Recommended experiments:
#   malignant_weight = 2.0 first
#   malignant_weight = 1.5 if FP increases too much
#   malignant_weight = 2.5 only if 2.0 is too weak
#
# Do not overwrite the original successful model:
#   best_multimodal_model_dual_roi_ce_bs4_seed42_init_mmibc_rerun.pth

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
# MMIBC-main/src/training/dinov2/dual_roi_fusion/train_multimodal_dual_roi_classweight_tunable_launcher.py
# parents[4] = MMIBC-main
PROJECT_ROOT = THIS_FILE.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Global class weight values
# ============================================================

_ORIGINAL_TORCH_NN_CE = nn.CrossEntropyLoss

GLOBAL_BENIGN_WEIGHT = 1.0
GLOBAL_MALIGNANT_WEIGHT = 2.0


class DynamicClassWeightedCrossEntropyLoss(nn.Module):
    """
    Tunable CrossEntropyLoss.

    class 0: benign
    class 1: malignant

    Weight tensor is automatically moved to the same device as logits.
    """

    def __init__(
        self,
        benign_weight=1.0,
        malignant_weight=2.0,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ):
        super().__init__()

        self.register_buffer(
            "class_weights",
            torch.tensor(
                [float(benign_weight), float(malignant_weight)],
                dtype=torch.float32,
            ),
        )

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Compatibility with older PyTorch signatures.
        self.size_average = size_average
        self.reduce = reduce

        print("=" * 80)
        print("TUNABLE CLASS WEIGHT MODE ENABLED")
        print("Using weighted CrossEntropyLoss")
        print(f"class 0 benign     weight = {float(benign_weight)}")
        print(f"class 1 malignant weight = {float(malignant_weight)}")
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

    If the source script calls:
        nn.CrossEntropyLoss()

    this launcher returns:
        DynamicClassWeightedCrossEntropyLoss(
            benign_weight=GLOBAL_BENIGN_WEIGHT,
            malignant_weight=GLOBAL_MALIGNANT_WEIGHT
        )

    If the source script already passes its own weight, we do not override it.
    """

    if "weight" in kwargs and kwargs["weight"] is not None:
        print("=" * 80)
        print("WARNING: Source script already passed weight to CrossEntropyLoss.")
        print("Using source script's own weight instead of tunable launcher.")
        print("=" * 80)
        return _ORIGINAL_TORCH_NN_CE(*args, **kwargs)

    if len(args) >= 1 and args[0] is not None:
        print("=" * 80)
        print("WARNING: Source script passed positional weight to CrossEntropyLoss.")
        print("Using source script's own weight instead of tunable launcher.")
        print("=" * 80)
        return _ORIGINAL_TORCH_NN_CE(*args, **kwargs)

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
        benign_weight=GLOBAL_BENIGN_WEIGHT,
        malignant_weight=GLOBAL_MALIGNANT_WEIGHT,
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
    global GLOBAL_BENIGN_WEIGHT
    global GLOBAL_MALIGNANT_WEIGHT

    parser = argparse.ArgumentParser(
        description=(
            "Run existing Dual-ROI MMIBC training script with tunable "
            "class-weighted CrossEntropyLoss."
        ),
        add_help=True,
    )

    parser.add_argument(
        "--source_script",
        type=str,
        required=True,
        help=(
            "Path to existing Dual-ROI training script. "
            "For the original ROI model, use train_multimodal_dual_roi.py."
        ),
    )

    parser.add_argument(
        "--benign_weight",
        type=float,
        default=1.0,
        help="Class weight for benign class 0.",
    )

    parser.add_argument(
        "--malignant_weight",
        type=float,
        default=2.0,
        help="Class weight for malignant class 1.",
    )

    launcher_args, source_args = parser.parse_known_args()

    if launcher_args.benign_weight <= 0:
        raise ValueError("--benign_weight must be positive.")

    if launcher_args.malignant_weight <= 0:
        raise ValueError("--malignant_weight must be positive.")

    GLOBAL_BENIGN_WEIGHT = float(launcher_args.benign_weight)
    GLOBAL_MALIGNANT_WEIGHT = float(launcher_args.malignant_weight)

    source_script_path = resolve_path(launcher_args.source_script)

    if not source_script_path.exists():
        raise FileNotFoundError(
            f"Source training script not found: {source_script_path}"
        )

    print("=" * 80)
    print("DUAL-ROI TUNABLE CLASS WEIGHT LAUNCHER")
    print("=" * 80)
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Source script    : {source_script_path}")
    print(f"Benign weight    : {GLOBAL_BENIGN_WEIGHT}")
    print(f"Malignant weight : {GLOBAL_MALIGNANT_WEIGHT}")
    print("Original source script will be executed without modifying the file.")
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

    runpy.run_path(str(source_script_path), run_name="__main__")


if __name__ == "__main__":
    main()