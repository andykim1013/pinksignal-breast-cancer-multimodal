import os
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.training.dinov2.multimodal_model.multimodal_dataset import get_advanced_medical_transforms


class MultimodalROIDataset(Dataset):
    """
    Multimodal dataset that uses:
        - mammography whole image
        - ultrasound ROI crop image

    Required CSV columns:
        mammo_path
        ultrasound_path
        ultrasound_roi_path
        label
        split
    """

    def __init__(self, csv_file, split="train", image_size=224, roi_col="ultrasound_roi_path"):
        logging.info(f"Loading ROI multimodal metadata from: {csv_file} for split: '{split}'")

        self.metadata = pd.read_csv(csv_file)
        self.split_df = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

        self.roi_col = roi_col
        self.label_map = {"benign": 0, "malignant": 1}

        required_cols = ["mammo_path", "ultrasound_path", "label", "split", self.roi_col]
        for col in required_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"Required column missing from ROI CSV: {col}")

        train_transforms, eval_transforms = get_advanced_medical_transforms(image_size)
        self.transform = train_transforms if split == "train" else eval_transforms

        logging.info(f"Loaded {len(self.split_df)} ROI pairs for split '{split}'")

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]

        mammo_path = row["mammo_path"]
        us_roi_path = row[self.roi_col]
        label_str = row["label"]

        if label_str not in self.label_map:
            raise ValueError(f"Unsupported label: {label_str}")

        if not os.path.exists(mammo_path):
            raise FileNotFoundError(f"Mammography image not found: {mammo_path}")

        if not os.path.exists(us_roi_path):
            raise FileNotFoundError(f"Ultrasound ROI image not found: {us_roi_path}")

        mammo_img = Image.open(mammo_path).convert("RGB")
        us_roi_img = Image.open(us_roi_path).convert("RGB")

        mammo_tensor = self.transform(mammo_img)
        us_roi_tensor = self.transform(us_roi_img)

        label = self.label_map[label_str]

        return mammo_tensor, us_roi_tensor, torch.tensor(label, dtype=torch.long)