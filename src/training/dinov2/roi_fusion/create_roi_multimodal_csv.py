import os
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "create_roi_multimodal_csv.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    return log_path


def normalize_path(path):
    return Path(str(path).replace("\\", os.sep).replace("/", os.sep))


def infer_masks_root(root):
    return Path(root) / "mmibc" / "ultrasound" / "masks"


def find_mask_files(us_path, masks_root):
    """
    Find BUSI mask file(s) corresponding to an ultrasound image.

    Expected examples:
        image: .../ultrasound/images/train/benign/benign (1).png
        mask : .../ultrasound/masks/train/benign/benign (1)_mask.png

    This function tries:
        1) Replace /images/ with /masks/
        2) stem + '_mask*'
        3) recursive fallback search under masks_root
    """
    us_path = normalize_path(us_path)
    masks_root = normalize_path(masks_root)

    stem = us_path.stem
    suffix = us_path.suffix

    candidates = []

    parts = list(us_path.parts)

    # Find "images" segment and replace with "masks"
    if "images" in parts:
        idx = parts.index("images")
        mask_parts = parts.copy()
        mask_parts[idx] = "masks"
        direct_mask_dir = Path(*mask_parts[:-1])

        candidates.extend(list(direct_mask_dir.glob(f"{stem}_mask*{suffix}")))
        candidates.extend(list(direct_mask_dir.glob(f"{stem}*mask*{suffix}")))
        candidates.extend(list(direct_mask_dir.glob(f"{stem}{suffix}")))

    # Fallback: search under masks root recursively
    if len(candidates) == 0 and masks_root.exists():
        candidates.extend(list(masks_root.rglob(f"{stem}_mask*{suffix}")))
        candidates.extend(list(masks_root.rglob(f"{stem}*mask*{suffix}")))

    # Remove duplicates while preserving order
    unique = []
    seen = set()
    for p in candidates:
        p = Path(p)
        if p.exists() and str(p) not in seen:
            unique.append(p)
            seen.add(str(p))

    return unique


def get_union_bbox_from_masks(mask_paths, image_size, threshold=10, margin_ratio=0.15):
    """
    Create union bounding box from one or more mask images.

    Returns:
        bbox: (x1, y1, x2, y2) or None
    """
    width, height = image_size
    union_mask = np.zeros((height, width), dtype=np.uint8)

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        if mask.shape[1] != width or mask.shape[0] != height:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        binary = (mask > threshold).astype(np.uint8)
        union_mask = np.maximum(union_mask, binary)

    ys, xs = np.where(union_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1

    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width - 1, x2 + margin_x)
    y2 = min(height - 1, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def save_roi_image(us_path, mask_paths, save_path, margin_ratio=0.15):
    """
    Save ROI crop image.

    If mask is missing or invalid, save original image as fallback.
    This keeps row counts identical to the original multimodal CSV.
    """
    us_path = normalize_path(us_path)
    save_path = normalize_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(us_path).convert("RGB")
    width, height = img.size

    status = "roi_crop"
    bbox = None

    if mask_paths:
        bbox = get_union_bbox_from_masks(mask_paths, (width, height), margin_ratio=margin_ratio)

    if bbox is None:
        crop = img
        status = "fallback_full_image"
    else:
        x1, y1, x2, y2 = bbox
        crop = img.crop((x1, y1, x2 + 1, y2 + 1))

    crop.save(save_path)

    return status, bbox


def main(args):
    log_file = setup_logging(args.output_dir)

    input_csv = normalize_path(args.input_csv)
    output_csv = normalize_path(args.output_csv)
    roi_root = normalize_path(args.roi_root)
    masks_root = normalize_path(args.masks_root)

    logging.info("Starting ROI CSV generation")
    logging.info(f"Input CSV: {input_csv}")
    logging.info(f"Output CSV: {output_csv}")
    logging.info(f"ROI root: {roi_root}")
    logging.info(f"Masks root: {masks_root}")
    logging.info(f"Margin ratio: {args.margin}")
    logging.info(f"Log file: {log_file}")

    df = pd.read_csv(input_csv)

    required_cols = ["mammo_path", "ultrasound_path", "label", "split"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing from input CSV: {col}")

    records = []
    missing_mask_rows = []
    fallback_count = 0
    roi_count = 0

    for idx, row in df.iterrows():
        mammo_path = row["mammo_path"]
        us_path = row["ultrasound_path"]
        label = row["label"]
        split = row["split"]

        us_path_obj = normalize_path(us_path)

        if not us_path_obj.exists():
            logging.warning(f"[{idx}] Ultrasound image missing: {us_path_obj}")
            missing_mask_rows.append({
                "index": idx,
                "reason": "ultrasound_missing",
                "ultrasound_path": str(us_path_obj),
            })
            continue

        mask_paths = find_mask_files(us_path_obj, masks_root)

        if len(mask_paths) == 0:
            missing_mask_rows.append({
                "index": idx,
                "reason": "mask_missing",
                "ultrasound_path": str(us_path_obj),
            })

        safe_stem = us_path_obj.stem.replace(" ", "_").replace("(", "").replace(")", "")
        roi_filename = f"{idx:06d}_{safe_stem}_roi.png"
        roi_save_path = roi_root / str(split) / str(label) / roi_filename

        status, bbox = save_roi_image(
            us_path=us_path_obj,
            mask_paths=mask_paths,
            save_path=roi_save_path,
            margin_ratio=float(args.margin),
        )

        if status == "roi_crop":
            roi_count += 1
        else:
            fallback_count += 1

        new_row = row.to_dict()
        new_row["ultrasound_mask_path"] = ";".join([str(p) for p in mask_paths])
        new_row["ultrasound_roi_path"] = str(roi_save_path)
        new_row["roi_status"] = status
        new_row["roi_bbox"] = "" if bbox is None else ",".join(map(str, bbox))

        records.append(new_row)

        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1} / {len(df)} rows")

    out_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    missing_df = pd.DataFrame(missing_mask_rows)
    missing_path = normalize_path(args.output_dir) / "roi_missing_or_problem_rows.csv"
    missing_df.to_csv(missing_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(out_df)),
        "roi_crop_count": int(roi_count),
        "fallback_full_image_count": int(fallback_count),
        "problem_rows": int(len(missing_df)),
        "output_csv": str(output_csv),
        "missing_report": str(missing_path),
    }

    summary_path = normalize_path(args.output_dir) / "roi_csv_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    logging.info("ROI CSV generation finished")
    logging.info(summary)

    print("\n" + "=" * 80)
    print("ROI CSV GENERATION COMPLETE")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ROI ultrasound images and ROI multimodal CSV.")

    parser.add_argument(
        "--root",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/multimodal_pairs_no_mammo_leak.csv",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/multimodal_pairs_roi_no_mammo_leak.csv",
    )

    parser.add_argument(
        "--roi_root",
        type=str,
        default="data/ultrasound_roi/images",
    )

    parser.add_argument(
        "--masks_root",
        type=str,
        default="data/ultrasound/masks",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/roi_fusion/roi_csv_generation",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Margin ratio around mask bounding box.",
    )

    args = parser.parse_args()
    main(args)