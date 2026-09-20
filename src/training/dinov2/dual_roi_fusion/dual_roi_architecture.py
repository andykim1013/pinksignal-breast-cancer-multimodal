import logging

import torch
import torch.nn as nn

from src.training.dinov2.unimodal_model.unimodal_model import DinoV2Classifier
from src.training.dinov2.multimodal_model.multimodal_architecture import (
    CrossAttention,
    GatedMultimodalUnit,
    ResidualBlock,
)


class DualUSContextFusion(nn.Module):
    """
    Fuse original ultrasound feature and ROI ultrasound feature.

    This block keeps both:
        - whole-image ultrasound context
        - lesion-centered ROI detail

    Input:
        original_us_features: [B, D]
        roi_us_features: [B, D]

    Output:
        enhanced_us_features: [B, D]
    """

    def __init__(self, feature_dim, dropout_rate=0.3):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.gate = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, original_us_features, roi_us_features):
        concat_features = torch.cat([original_us_features, roi_us_features], dim=-1)

        fused = self.fusion(concat_features)

        gate = torch.sigmoid(self.gate(concat_features))

        enhanced_us_features = gate * fused + (1.0 - gate) * original_us_features

        return enhanced_us_features


class DualROIMultimodalFusionModel(nn.Module):
    """
    4-1 Dual-ROI MMIBC model.

    Inputs:
        mammo_x: mammography whole image
        us_x: original ultrasound whole image
        us_roi_x: ultrasound ROI crop image

    Structure:
        Mammo image -> mammo DINOv2 encoder -> mammo feature
        Original US image -> shared US DINOv2 encoder -> original US feature
        ROI US image -> shared US DINOv2 encoder -> ROI US feature

        original US feature + ROI US feature -> DualUSContextFusion
        Mammo feature + enhanced US feature -> Cross-attention + gated fusion
        classifier -> benign/malignant
    """

    def __init__(
        self,
        unimodal_config,
        us_model_path,
        mammo_model_path,
        dropout_rate=0.3,
    ):
        super().__init__()

        logging.info("Loading pre-trained unimodal encoders for Dual-ROI model.")

        self.us_encoder = DinoV2Classifier(
            n_classes=unimodal_config["model"]["n_classes"],
            model_name=unimodal_config["model"]["name"],
            dropout_rate=unimodal_config["training"]["dropout_rate"],
        )
        self.us_encoder.load_state_dict(torch.load(us_model_path, map_location="cpu"))

        self.mammo_encoder = DinoV2Classifier(
            n_classes=unimodal_config["model"]["n_classes"],
            model_name=unimodal_config["model"]["name"],
            dropout_rate=unimodal_config["training"]["dropout_rate"],
        )
        self.mammo_encoder.load_state_dict(torch.load(mammo_model_path, map_location="cpu"))

        feature_dim = self.us_encoder.backbone.embed_dim

        logging.info(f"Dual-ROI feature dimension: {feature_dim}")

        self.us_context_fusion = DualUSContextFusion(
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
        )

        self.mammo_attends_to_us = CrossAttention(feature_dim)
        self.us_attends_to_mammo = CrossAttention(feature_dim)

        self.gated_fusion = GatedMultimodalUnit(input_dim=feature_dim)

        self.fusion_classifier = nn.Sequential(
            ResidualBlock(feature_dim, 1024),
            ResidualBlock(1024, 512),
            nn.Linear(512, 2),
        )

        # This flag is controlled in train script.
        # If encoders are frozen, we use no_grad for lower memory usage.
        self.freeze_encoders = False

        logging.info("Dual-ROI multimodal model initialized successfully.")

    def _extract_backbone_features(self, encoder, x):
        if self.freeze_encoders:
            with torch.no_grad():
                return encoder.backbone(x)
        return encoder.backbone(x)

    def forward(self, mammo_x, us_x, us_roi_x):
        mammo_features = self._extract_backbone_features(self.mammo_encoder, mammo_x)

        original_us_features = self._extract_backbone_features(self.us_encoder, us_x)
        roi_us_features = self._extract_backbone_features(self.us_encoder, us_roi_x)

        enhanced_us_features = self.us_context_fusion(
            original_us_features,
            roi_us_features,
        )

        mammo_attended = self.mammo_attends_to_us(
            mammo_features,
            enhanced_us_features,
        )

        us_attended = self.us_attends_to_mammo(
            enhanced_us_features,
            mammo_features,
        )

        fused_features = self.gated_fusion(
            mammo_attended,
            us_attended,
        )

        output = self.fusion_classifier(fused_features)

        return output