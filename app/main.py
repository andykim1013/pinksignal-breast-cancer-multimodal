"""
BreastAI FastAPI Backend
========================
DualROIMultimodalFusionModel (DINOv2 기반) 연동 버전

실행 방법 (저장소 루트에서):
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

[공개 저장소 안내]
원본 파일은 개인 PC 절대경로(C:/Users/<username>/Desktop/MMIBC-main)를
MMIBC_ROOT로 하드코딩하고 있었습니다. 공개 저장소에서는 이 파일(app/main.py)의
위치를 기준으로 저장소 루트를 자동 계산하도록 아래 "경로 설정" 블록만 수정했으며,
그 외 로직은 원본과 동일합니다. 환경변수 MMIBC_ROOT를 지정하면 다른 위치의
saved_models/ 등을 사용할 수도 있습니다.
"""

import io
import os
import sys
import base64
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────
# 경로 설정 — 저장소 루트를 import 경로에 추가
# (수정된 부분: 기존 개발환경에 포함되어 있던 개인 PC 절대경로를 제거하고,
#  이 파일 위치 기준 저장소 루트 자동 계산 + MMIBC_ROOT 환경변수로 대체)
# ──────────────────────────────────────────────────────────────
MMIBC_ROOT = Path(os.environ.get("MMIBC_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(MMIBC_ROOT))

# ──────────────────────────────────────────────────────────────
# 팀 모델 파일 경로
# ──────────────────────────────────────────────────────────────
MULTIMODAL_MODEL_PATH = MMIBC_ROOT / "saved_models/best_multimodal_model_dual_roi_ce_bs4_seed42_init_mmibc_rerun.pth"
US_MODEL_PATH         = MMIBC_ROOT / "saved_models/best_model_final.pth"
MAMMO_MODEL_PATH      = MMIBC_ROOT / "saved_models/best_mammo_model_final.pth"
UNIMODAL_CONFIG_PATH  = MMIBC_ROOT / "src/training/dinov2/unimodal_model/config.yaml"

INPUT_SIZE = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("breastai")
logger.info(f"Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# 전처리 (DINOv2 표준)
# ──────────────────────────────────────────────────────────────
import cv2

# ──────────────────────────────────────────────────────────────
# 전처리 (팀 학습 코드 multimodal_dataset.py 의 eval_transforms와 동일하게 맞춤)
# CLAHE -> Resize -> ToTensor -> Normalize
# ──────────────────────────────────────────────────────────────
class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img):
        img_np = np.array(img.convert('L'))
        cl_img = self.clahe.apply(img_np)
        return Image.fromarray(cl_img).convert('RGB')

preprocess = transforms.Compose([
    ApplyCLAHE(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ──────────────────────────────────────────────────────────────
# 팀 모델 로딩
# ──────────────────────────────────────────────────────────────
class ModelWrapper:
    def __init__(self):
        self.model = None
        self.ready = False
        self._load()

    def _load(self):
        try:
            import yaml
            from src.training.dinov2.dual_roi_fusion.dual_roi_architecture import DualROIMultimodalFusionModel

            logger.info("unimodal config 로딩 중...")
            with open(UNIMODAL_CONFIG_PATH, "r") as f:
                unimodal_config = yaml.safe_load(f)

            logger.info("DualROIMultimodalFusionModel 구조 초기화 중...")
            model = DualROIMultimodalFusionModel(
                unimodal_config  = unimodal_config,
                us_model_path    = str(US_MODEL_PATH),
                mammo_model_path = str(MAMMO_MODEL_PATH),
            )

            logger.info(f"가중치 로딩 중: {MULTIMODAL_MODEL_PATH}")
            state_dict = torch.load(MULTIMODAL_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)

            model.eval().to(DEVICE)
            self.model = model
            self.ready = True
            logger.info("✅ 팀 DualROI 모델 로드 완료!")

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}", exc_info=True)
            logger.warning("🔶 데모 모드로 실행 (랜덤 가중치) — 실제 진단 불가")
            self._load_demo()

    def _load_demo(self):
        """모델 로드 실패 시 더미 모델로 대체"""
        from torchvision import models as tvm

        class DemoWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                base = tvm.resnet18(weights=None)
                base.fc = nn.Linear(base.fc.in_features, 2)
                self.net = base
            def forward(self, mammo_x, us_x, us_roi_x):
                # 3개 입력을 평균 내서 단일 모델에 통과
                x = (mammo_x + us_x + us_roi_x) / 3.0
                return self.net(x)

        self.model = DemoWrapper().eval().to(DEVICE)
        self.ready = False


model_wrapper = ModelWrapper()

# ──────────────────────────────────────────────────────────────
# Attention Rollout 기반 Grad-CAM
# DINOv2는 CNN이 아니라 ViT라서 마지막 attention map을 사용합니다.
# ──────────────────────────────────────────────────────────────
class DinoAttentionMap:
    """
    DINOv2 backbone의 마지막 레이어 attention을 이용한 히트맵 생성.
    DinoV2Classifier.get_attention_maps() 활용.
    """
    def get(self, encoder, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Returns:
            heatmap: float32 ndarray shape (INPUT_SIZE, INPUT_SIZE), 0~1
        """
        try:
            attn = encoder.get_attention_maps(img_tensor)
            # attn shape: (batch, num_heads, num_patches+1, num_patches+1)
            # CLS token → patch attention (첫 행, CLS 제외)
            attn_cls = attn[0, :, 0, 1:]          # (num_heads, num_patches)
            attn_mean = attn_cls.mean(0)            # (num_patches,)

            # patch grid 크기 계산 (DINOv2 patch_size=14)
            patch_size = 14
            grid = INPUT_SIZE // patch_size         # 16 for 224px
            attn_map = attn_mean[:grid*grid].reshape(grid, grid)

            # 0~1 정규화
            a_min, a_max = attn_map.min(), attn_map.max()
            if a_max > a_min:
                attn_map = (attn_map - a_min) / (a_max - a_min)

            # 원본 크기로 업샘플링
            attn_np  = attn_map.cpu().float().numpy()
            attn_pil = Image.fromarray((attn_np * 255).astype(np.uint8))
            attn_res = attn_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            return np.array(attn_res, dtype=np.float32) / 255.0

        except Exception as e:
            logger.warning(f"Attention map 생성 실패: {e}")
            return np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)


attn_map_gen = DinoAttentionMap()

# ──────────────────────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────────────────────
def decode_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def probs_to_birads(p: float) -> str:
    if   p < 0.05: return "1"
    elif p < 0.15: return "2"
    elif p < 0.35: return "3"
    elif p < 0.55: return "4a"
    elif p < 0.70: return "4b"
    elif p < 0.85: return "4c"
    elif p < 0.95: return "5"
    else:          return "6"

def generate_finding(pct: int, birads: str):
    if pct < 15:
        return (
            "영상에서 뚜렷한 악성 소견이 관찰되지 않습니다. 정기적인 추적 관찰을 권장합니다.",
            "명확한 경계의 등에코 결절 또는 정상 실질",
            "명확 (circumscribed)",
            "측정불가 (이상 소견 없음)",
            "정기 검진 유지 (12개월 후 추적)",
            None
        )
    elif pct < 35:
        return (
            "초음파에서 경계가 비교적 명확한 저에코 결절이 관찰됩니다. 양성 병변 가능성이 높으나 단기 추적이 필요합니다.",
            "타원형 저에코 결절",
            "비교적 명확 (indistinct)",
            "~8–12 mm 추정",
            "6개월 후 초음파 추적 검사 권고",
            "필요 시 세침흡인세포검사(FNA) 고려"
        )
    elif pct < 60:
        return (
            "초음파 및 맘모그래피에서 경계가 불분명한 저에코 결절이 확인됩니다. 악성 가능성을 배제할 수 없어 조직 생검이 권장됩니다.",
            "불규칙형 저에코 결절, 후방 음향 감소",
            "불분명 (indistinct) / 침상형 (spiculated)",
            "~15–20 mm 추정",
            "초음파 유도하 core needle biopsy 시행 권고",
            "MRI 추가 검사 고려 가능"
        )
    else:
        return (
            "맘모그래피 및 초음파 모두에서 고악성 의심 소견이 확인됩니다. 침상형 경계와 후방 음향 감소를 동반한 불규칙형 결절이 관찰됩니다.",
            "불규칙형 저에코 결절, 침상형 경계, 후방 음향 감소",
            "침상형 (spiculated)",
            "~20–30 mm 추정",
            "즉각적인 조직 생검 및 다학제 진료 회의(MDT) 권고",
            "림프절 전이 여부 확인을 위한 추가 영상 검사 고려"
        )

# ──────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="BreastAI API",
    description="유방 영상 AI 분석 서버 — DualROI DINOv2 모델 연동",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    mmg:        str
    us:         str
    roi:        str
    patient_id: Optional[str] = "UNKNOWN"

class AnalysisResponse(BaseModel):
    patient_id:     str
    timestamp:      str
    model_ready:    bool
    birads:         str
    malignancy_pct: int
    finding:        str
    size_estimate:  str
    morphology:     str
    margin:         str
    recommendation: str
    note:           Optional[str]
    gradcam_mmg:    list[float]
    gradcam_us:     list[float]
    gradcam_roi:    list[float]
    gradcam_w:      int
    gradcam_h:      int

@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "model_ready":  model_wrapper.ready,
        "device":       str(DEVICE),
        "model_path":   str(MULTIMODAL_MODEL_PATH),
        "timestamp":    datetime.now().isoformat()
    }

@app.post("/api/process_breast", response_model=AnalysisResponse)
def process_breast(req: AnalysisRequest):
    try:
        logger.info(f"분석 요청: {req.patient_id}")

        # 1) 이미지 디코딩 + 텐서 변환
        img_mmg = decode_image(req.mmg)
        img_us  = decode_image(req.us)
        img_roi = decode_image(req.roi)

        t_mmg = preprocess(img_mmg).unsqueeze(0).to(DEVICE)
        t_us  = preprocess(img_us ).unsqueeze(0).to(DEVICE)
        t_roi = preprocess(img_roi).unsqueeze(0).to(DEVICE)

        # 2) 추론 — DualROIMultimodalFusionModel.forward(mammo_x, us_x, us_roi_x)
        #    + Saliency Map (입력 픽셀 기준 gradient) 동시 계산
        t_mmg.requires_grad_(True)
        t_us.requires_grad_(True)
        t_roi.requires_grad_(True)

        model_wrapper.model.zero_grad(set_to_none=True)
        logits = model_wrapper.model(t_mmg, t_us, t_roi)  # (1, 2)
        probs  = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        malignancy_prob = float(probs[1])
        malignancy_pct  = int(round(malignancy_prob * 100))
        birads          = probs_to_birads(malignancy_prob)
        logger.info(f"추론 완료 — malignancy: {malignancy_pct}%, BI-RADS: {birads}")

        # 3) Saliency map 생성 (악성 클래스 점수에 대한 gradient)
        gcam_mmg = gcam_us = gcam_roi = []
        ENABLE_SALIENCY_MAP = True
        if ENABLE_SALIENCY_MAP and model_wrapper.ready:
            try:
                score = logits[0, 1]  # malignant class score
                score.backward()

                def grad_to_contour(tensor):
                    """
                    Gradient(saliency) 맵에서 가장 활성화된 영역의
                    윤곽선(contour)을 정규화 좌표(0~1) 리스트로 반환.
                    [x1,y1, x2,y2, x3,y3, ...]
                    """
                    grad = tensor.grad
                    if grad is None:
                        return []

                    sal = grad.abs().sum(dim=1).squeeze(0)  # (H, W)
                    sal = sal.detach().cpu().numpy().astype(np.float32)

                    # 약하게 블러 → 노이즈 제거 (윤곽선 추출 안정화)
                    sal = cv2.GaussianBlur(sal, (15, 15), 0)

                    s_min, s_max = sal.min(), sal.max()
                    if s_max <= s_min:
                        return []
                    sal = (sal - s_min) / (s_max - s_min)

                    # 상위 활성화 영역 마스크 (percentile 기반 — 분포에 따라 자동 조정)
                    thresh = np.percentile(sal, 80)
                    mask = (sal >= thresh).astype(np.uint8)

                    # 작은 노이즈 제거 + 영역 채우기
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        return []

                    # 가장 큰 영역 선택
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) < 15:
                        return []

                    # 점 개수 줄이기 (너무 각지지 않게, 모양은 유지)
                    epsilon = 0.004 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)

                    if len(approx) < 3:
                        approx = c

                    pts_arr = approx.reshape(-1, 2)
                    # 점이 너무 많으면 균등 샘플링
                    MAX_PTS = 60
                    if len(pts_arr) > MAX_PTS:
                        idx = np.linspace(0, len(pts_arr) - 1, MAX_PTS).astype(int)
                        pts_arr = pts_arr[idx]

                    H, W = sal.shape
                    points = []
                    for pt in pts_arr:
                        points.append(float(pt[0]) / W)
                        points.append(float(pt[1]) / H)
                    return points

                gcam_mmg = grad_to_contour(t_mmg)
                gcam_us  = grad_to_contour(t_us)
                gcam_roi = grad_to_contour(t_roi)

            except Exception as e:
                logger.warning(f"Saliency contour 생성 실패 (빈 윤곽선 반환): {e}")
                gcam_mmg = gcam_us = gcam_roi = []
        else:
            gcam_mmg = gcam_us = gcam_roi = []
        # gcam_*는 윤곽선 좌표 [x1,y1,x2,y2,...] (정규화 0~1). 프론트엔드에서 선으로 그립니다.

        # 4) 텍스트 소견
        finding, morphology, margin, size_est, recommendation, note = \
            generate_finding(malignancy_pct, birads)

        logger.info(f"완료: {req.patient_id} — BI-RADS {birads} ({malignancy_pct}%)")

        return AnalysisResponse(
            patient_id     = req.patient_id,
            timestamp      = datetime.now().isoformat(),
            model_ready    = model_wrapper.ready,
            birads         = birads,
            malignancy_pct = malignancy_pct,
            finding        = finding,
            size_estimate  = size_est,
            morphology     = morphology,
            margin         = margin,
            recommendation = recommendation,
            note           = note,
            gradcam_mmg    = gcam_mmg,
            gradcam_us     = gcam_us,
            gradcam_roi    = gcam_roi,
            gradcam_w      = INPUT_SIZE,
            gradcam_h      = INPUT_SIZE,
        )

    except Exception as e:
        logger.error(f"분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
