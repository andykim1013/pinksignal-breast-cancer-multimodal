# PINKSIGNAL — Breast Cancer Multimodal Classification (MMIBC)

> ⚠️ **연구/교육 목적 프로토타입 안내**
> 본 프로젝트는 대학교 캡스톤디자인(4학년 1학기 설계) 과정에서 진행된 **연구 및 교육 목적의 프로토타입**이며, 실제 임상 진단을 위한 의료기기가 아닙니다. 이 저장소의 코드나 데모의 출력 결과를 실제 진단·치료 판단에 사용해서는 안 됩니다.

## 프로젝트 소개

**PINKSIGNAL**(프로젝트명 "소노핑크")은 맘모그래피(mammography)와 초음파(ultrasound) 영상을 함께 사용하는 **멀티모달 딥러닝 모델(MMIBC, Multi-Modal breast cancer Image-Based Classification)** 로 유방 병변의 양성/악성을 분류하는 연구 프로젝트입니다. 여기서 한 걸음 더 나아가, 초음파 원본 영상 외에 병변 부위를 잘라낸 **ROI(Region of Interest) 영상**을 세 번째 입력으로 추가한 **Triple-Input(Dual-ROI) 구조**를 제안·구현했습니다.

## 연구 목적

- 국내 유방암은 45–74세에서 다발하며, 검진 수검률이 계층별로 큰 격차(전체 64.6% vs. 장애인 46.1%, 의료급여수급권자 28.9%)를 보이고, 기존 국가검진 판독 성능이 권장 기준(BI-RADS 기준 민감도 등)에 못 미치는 사례가 보고되어 있습니다.
- 맘모그래피 또는 초음파 **단일 모달리티** 판독은 각각 한계가 있어(예: 치밀유방에서 맘모그래피 민감도 저하), 두 모달리티를 함께 활용하면 진단 보조 성능을 높일 수 있다는 가설에서 출발했습니다.
- 특히 기존 2-Input 멀티모달 모델은 초음파 원본 영상만 사용해 병변의 형태·경계·내부 패턴 같은 **병변 중심 정보**가 충분히 강조되지 않을 수 있다는 한계가 있어, 이를 보완하기 위해 **악성 재현율(Malignant Recall) 개선**을 핵심 목표로 두었습니다.

## 주요 기능

- DINOv2(ViT) 백본 기반 **유니모달 분류기**(초음파 전용 / 맘모그래피 전용)
- Cross-Attention + Gated Multimodal Fusion 기반 **2-Input 멀티모달 모델(MMIBC)**
- 초음파 원본 + 초음파 ROI + 맘모그래피를 함께 사용하는 **Triple-Input(Dual-ROI) 확장 모델**
- Class-weighted Cross-Entropy, Focal Loss, Class-Balanced Focal Loss 등 **재현율(recall) 개선 실험 코드**
- 이미지 업로드 → AI 추론 → 결과 표시를 시연하는 **FastAPI 웹 데모**

## 모델 구조

### 2-Input MMIBC (기존 구조)

```
Mammography Image ──▶ DINOv2 encoder ──▶ mammo feature ─┐
                                                          ├─▶ Cross-Attention ─▶ Gated Fusion ─▶ Classifier ─▶ Benign / Malignant
Ultrasound Original Image ──▶ DINOv2 encoder ──▶ us feature ─┘
```

`src/training/dinov2/multimodal_model/multimodal_architecture.py`의 `MultimodalFusionModel`이 이 구조를 구현합니다. 두 인코더(`DinoV2Classifier`, `src/training/dinov2/unimodal_model/unimodal_model.py`)는 각각 유니모달 학습으로 미리 학습된 가중치를 불러온 뒤, `CrossAttention`으로 서로의 특징에 주목하고 `GatedMultimodalUnit`으로 동적 가중 융합한 뒤 `ResidualBlock` 기반 분류기로 양성/악성을 분류합니다.

### Triple-Input(Dual-ROI) 확장 구조

```
Mammography Image ──────────────▶ DINOv2 encoder (mammo) ──▶ mammo feature ─┐
Ultrasound Original Image ─┐                                                 │
                            ├─▶ 공유 DINOv2 encoder (us) ─▶ DualUSContextFusion ─▶ enhanced us feature ─▶ Cross-Attention ─▶ Gated Fusion ─▶ Classifier ─▶ Benign / Malignant
Ultrasound ROI Image ───────┘
```

`src/training/dinov2/dual_roi_fusion/dual_roi_architecture.py`의 `DualROIMultimodalFusionModel`이 이 구조를 구현합니다. 초음파 원본(문맥 정보)과 초음파 ROI(병변 중심 정보)를 같은 초음파 인코더로 각각 추출한 뒤 `DualUSContextFusion`으로 먼저 결합하고, 이후 맘모그래피 특징과 Cross-Attention + Gated Fusion으로 최종 융합합니다. **FastAPI 웹 데모(`app/main.py`)가 실제로 사용하는 모델이 바로 이 구조입니다.**

### DINOv2의 역할

두 구조 모두 각 모달리티의 특징 추출기로 Meta의 **DINOv2**(Vision Transformer 기반 self-supervised 사전학습 모델)를 `torch.hub.load('facebookresearch/dinov2', model_name, ...)`로 불러와 사용합니다(`src/training/dinov2/unimodal_model/unimodal_model.py`의 `DinoV2Classifier`). 학습 시 인터넷 연결이 필요하며(최초 실행 시 torch hub에서 가중치를 내려받음), 백본을 전부/일부 동결(freeze)한 뒤 fine-tuning 하는 progressive fine-tuning 방식도 구현되어 있습니다.

## 전체 처리 흐름

**학습(Training)**
```
VinDr-Mammo / BUSI 원본 이미지 + CSV 라벨
   ↓ (multimodal_dataset.py: CLAHE 전처리, synthetic pairing)
DataLoader (torchvision transforms)
   ↓
DINOv2 인코더 (Whole US / ROI US / Mammo)
   ↓
Cross-Attention + Gated Fusion
   ↓
Classifier → Benign / Malignant
   ↓ (AdamW + class-weighted/Focal Loss + ReduceLROnPlateau)
best_*.pth 저장 + TensorBoard / scikit-learn 평가지표
```

**추론(웹 데모)**
```
사용자가 웹(app/home.html)에서 맘모그래피 + 초음파 원본 + 초음파 ROI 이미지 업로드
   ↓
FastAPI(app/main.py) 수신 → CLAHE 전처리 → 224x224 정규화
   ↓
DualROIMultimodalFusionModel(.pth) 로드 → 추론 + saliency map 계산
   ↓
악성 확률 / BI-RADS 추정 / 소견 텍스트 반환 → home.html에 결과 표시
```

## 사용 기술

| 구분 | 내용 |
|---|---|
| 언어 | Python |
| 딥러닝 프레임워크 | PyTorch, torchvision, DINOv2(`torch.hub`) |
| 서빙 프레임워크 | FastAPI, Uvicorn |
| 주요 라이브러리 | pandas, NumPy, scikit-learn, OpenCV(CLAHE 전처리), Pillow, Matplotlib/Seaborn, TensorBoard, grad-cam(Class Activation Map), PyYAML |
| 데이터베이스 | 없음 (CSV 기반 라벨/메타데이터) |
| Frontend | 정적 HTML(`app/home.html`), 별도 SPA 프레임워크 없음 |
| Python / CUDA 버전 | 원본 프로젝트에 명시된 파일이 없어 **확인 필요** (개발 환경이 Windows 개인/랩 PC였다는 흔적만 확인됨) |

## Repository 구조

```
pinksignal-breast-cancer-multimodal/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   └── training/
│       └── dinov2/
│           ├── unimodal_model/        # 유니모달(초음파/맘모) 인코더, 학습/평가/XAI 스크립트
│           ├── multimodal_model/      # 2-Input MMIBC 모델, 학습/평가/XAI 스크립트
│           ├── dual_roi_fusion/       # Triple-Input(Dual-ROI) 모델 및 학습 스크립트 (웹 데모가 사용하는 구조)
│           ├── recall_optimization/   # Focal Loss / Class-Balanced Focal Loss 등 recall 개선 실험
│           ├── roi_fusion/            # ROI 단일 입력 fusion 실험
│           └── baselines/             # Weighted late fusion 등 비교 baseline
├── app/                                # FastAPI 웹 데모
│   ├── main.py
│   └── home.html
├── docs/
│   └── usecase_sequence_diagram.uml   # Use Case / Sequence 다이어그램 (StarUML)
├── data/                                # (비어 있음) 데이터셋을 직접 준비해서 배치하는 위치
└── saved_models/                       # (비어 있음) 학습된 .pth 가중치를 배치하는 위치
```

> **구조에 대한 설명**: `src/training/dinov2/` 하위 코드는 `from src.training.dinov2.multimodal_model.multimodal_architecture import ...` 형태의 절대 경로 import를 사용합니다. 이 import 관계를 깨뜨리지 않기 위해 원본의 폴더 구조(`unimodal_model/`, `multimodal_model/`, `dual_roi_fusion/`, `recall_optimization/`, `roi_fusion/`, `baselines/`)를 그대로 유지했습니다. 흔한 `src/` 최상위에 모델 폴더를 바로 두는 구조 대신 `src/training/dinov2/...` 3단계 구조를 쓰는 것도 같은 이유입니다.

## 설치 방법

```bash
git clone <this-repo-url>
cd pinksignal-breast-cancer-multimodal
pip install -r requirements.txt
```

DINOv2 백본은 최초 실행 시 `torch.hub`를 통해 인터넷에서 자동으로 내려받습니다.

## 데이터 준비 방법

이 저장소에는 **어떤 원본 데이터도 포함되어 있지 않습니다.** 아래 데이터셋을 직접 준비해서 `data/` 아래에 배치해야 합니다.

| 데이터셋 | 역할 | 비고 |
|---|---|---|
| VinDr-Mammo | 맘모그래피 학습 데이터 | 공개 학술 데이터셋. 공식 배포처에서 라이선스·이용조건을 확인 후 다운로드하세요. |
| BUSI (Breast Ultrasound Images) | 초음파 학습 데이터 (benign/malignant, normal 클래스 제외) | 공개 학술 데이터셋. 공식 배포처에서 라이선스·이용조건을 확인 후 다운로드하세요. |

> 데이터셋의 정확한 공식 배포 URL은 로컬 문서에서 명시적으로 확인되지 않아 이 README에는 임의로 링크를 적지 않았습니다. **다운로드 전 각 데이터셋의 공식 페이지에서 최신 라이선스/이용조건을 직접 확인해 주세요.**

`unimodal_model/config.yaml` 기준 예상 배치 구조:
```
data/
├── ultrasound/images/...
├── mammo/...
│   └── vindr_mammo_metadata.csv
└── multimodal_pairs.csv        # 맘모-초음파 synthetic pairing 결과 CSV (아래 참고)
```

`multimodal_pairs.csv`(및 Dual-ROI용 `multimodal_pairs_roi_margin030_no_mammo_leak.csv`)는 `src/training/dinov2/multimodal_model/pairing_data.py`가 라벨 기준으로 맘모그래피-초음파 쌍을 생성(synthetic pairing)한 결과물이며, `MultimodalDataset`(`multimodal_dataset.py`)이 학습 시 이 CSV를 읽어 이미지 쌍을 로드합니다. 즉 원본 이미지 데이터를 준비한 뒤, 이 스크립트로 CSV를 먼저 생성해야 멀티모달 학습이 가능합니다.

## 모델 weight 준비 방법

이 저장소에는 **어떤 `.pth` 가중치도 포함되어 있지 않습니다.** 직접 학습하거나(아래 "학습 방법" 참고) 팀 내부에서 별도로 전달받은 가중치 파일을 `saved_models/`에 배치해야 합니다. 코드가 참조하는 파일명은 다음과 같습니다.

| 파일명 | 역할 |
|---|---|
| `best_model_final.pth` | 유니모달 **초음파** 인코더 가중치 |
| `best_mammo_model_final.pth` | 유니모달 **맘모그래피** 인코더 가중치 |
| `best_multimodal_model.pth` | 2-Input MMIBC 멀티모달 모델 가중치 (`multimodal_train.py`가 저장하는 기본 파일명) |
| `best_multimodal_model_dual_roi_ce_bs4_seed42_init_mmibc_rerun.pth` | **웹 데모(`app/main.py`)가 실제로 로드하는 Triple-Input(Dual-ROI) 최종 모델** |

`train_multimodal_dual_roi_classweight3p0_launcher.py`, `train_multimodal_dual_roi_highres.py` 등은 실험 조건(class weight, 해상도 등)에 따라 다른 파일명으로 가중치를 저장하도록 되어 있어, 실험용 변형 가중치가 다수 존재할 수 있습니다. 위 4개 파일명이 "최종/데모용"으로 확인된 것이고, 나머지는 실험 로그 성격입니다.

## 학습 방법

모든 학습/평가 스크립트의 `argparse` 기본값은 **저장소 루트에서 실행한다는 전제로 상대경로**(`data/...`, `saved_models/...`, `outputs/...`, `src/training/dinov2/...`)로 맞춰져 있습니다. 즉 `data/`, `saved_models/`를 준비한 뒤 저장소 루트에서 그대로 실행하면 되며, 다른 위치를 쓰고 싶을 때만 아래처럼 인자를 직접 지정하면 됩니다.

```bash
# 1) 유니모달 인코더 학습 (초음파 / 맘모그래피 각각)
python src/training/dinov2/unimodal_model/ultrasound_train.py --config src/training/dinov2/unimodal_model/config.yaml
python src/training/dinov2/unimodal_model/mammo_train.py --config src/training/dinov2/unimodal_model/config.yaml

# 2) 2-Input MMIBC 멀티모달 학습 (위에서 학습한 유니모달 가중치 필요)
python src/training/dinov2/multimodal_model/multimodal_train.py --config src/training/dinov2/multimodal_model/config.yaml

# 3) Triple-Input(Dual-ROI) 학습 — 웹 데모가 사용하는 최종 구조
# (--root, --config, --roi_csv 등은 모두 기본값이 저장소 상대경로이므로, 필요할 때만 재지정)
python src/training/dinov2/dual_roi_fusion/train_multimodal_dual_roi.py \
  --root . \
  --config src/training/dinov2/multimodal_model/config_no_mammo_leak.yaml \
  --roi_csv data/multimodal_pairs_roi_margin030_no_mammo_leak.csv
```

> **참고 (`baselines/weighted_late_fusion_baseline.py`)**: 이 스크립트가 읽는 pairing CSV(`multimodal_pairs.csv`)에 다른(예전) 개발 환경에서 생성되어 그 환경의 절대경로가 그대로 저장되어 있는 경우를 대비한 `replace_old_root()` 보정 함수가 있습니다. **기본 공개 저장소는 어떤 개인 PC 절대경로에도 의존하지 않으며**, 이 기능은 완전히 선택 사항입니다. 새로 데이터를 준비하는 일반적인 사용자는 신경 쓸 필요가 없습니다. 과거에 다른 경로 체계로 생성된 CSV를 재사용해야 하는 경우에만, 환경변수 `MMIBC_LEGACY_ROOTS`에 옛 root 경로를 `;`로 구분해 지정하면 해당 접두사가 현재 `--root`로 자동 치환됩니다. 환경변수를 지정하지 않으면 이 함수는 아무 것도 바꾸지 않습니다.

## FastAPI 웹 데모 실행 방법

```bash
# 저장소 루트에서 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 `app/home.html`을 열면(또는 정적 파일 서빙 설정 시) 맘모그래피/초음파 원본/초음파 ROI 이미지를 업로드해 분석 결과를 확인할 수 있습니다. 기본적으로 `saved_models/` 아래의 4개 가중치 파일이 없으면 데모는 **랜덤 가중치의 더미 모델**로 대체 실행되며(`model_ready: false`), 이 경우 실제 진단 결과가 아님을 유의해야 합니다.

## 실험 결과

Triple-Input(Dual-ROI) 모델의 Validation set 기준 실험 결과입니다(출처: 팀 논문 초록, `라벨 기반 synthetic pairing + No-Mammo-Leak Clean Baseline` 조건):

| 지표 | 값 |
|---|---|
| Accuracy | 92.98% |
| Macro F1-score | 0.9029 |
| Weighted F1-score | 0.9280 |
| ROC AUC | 0.9533 |
| Malignant Recall | 0.7931 |

이 수치는 팀이 작성한 논문 초록에 기재된 **Validation set** 결과이며, 별도의 held-out test set에 대한 일반화 성능이 이 저장소 코드로 재검증된 것은 아닙니다. 성능을 인용할 때는 반드시 "Validation set 기준"임을 함께 명시해 주세요.

## 알려진 한계

- 위 실험 결과는 Validation set 기준이며, 독립적인 test set 성능은 별도로 확인되지 않았습니다.
- 학습에 사용된 맘모그래피-초음파 쌍은 동일 환자의 실제 페어가 아니라 **라벨 기준 synthetic pairing**으로 구성되어 있어, 실제 임상 페어 데이터에서의 성능은 다를 수 있습니다.
- 데이터 불균형(악성 비율이 낮음) 문제로 Malignant Recall이 다른 지표 대비 상대적으로 낮으며, 이를 개선하기 위한 Focal Loss / Class-Balanced Loss 실험이 진행 중입니다(`recall_optimization/`).
- 일부 학습 스크립트의 argparse 기본 경로가 아직 정리되지 않았습니다(위 "학습 방법" 참고).
- 웹 데모는 시연용 프로토타입이며, 의료기기 인허가나 임상 검증을 거치지 않았습니다.

## 데이터셋 및 외부 프로젝트 고지

- **VinDr-Mammo**, **BUSI**: 본 프로젝트가 사용한 공개 학술 데이터셋입니다. 데이터 자체는 이 저장소에 포함되어 있지 않으며, 각 데이터셋의 라이선스와 이용조건은 공식 배포처에서 확인해야 합니다.
- **DINOv2** (Meta AI, `facebookresearch/dinov2`): 특징 추출 백본으로 `torch.hub`를 통해 그대로 불러와 사용했습니다.
- 라이선스: 이 저장소 자체의 라이선스는 아직 지정되지 않았습니다(LICENSE 파일 없음, **확인 필요**). 코드를 재사용/배포하기 전에 팀과 라이선스를 협의해 주세요.

## 참고문헌

프로젝트 진행 중 조사한 관련 연구입니다(제목만 기재하며, 정확한 서지정보는 각 원문에서 확인해 주세요): LUMINA, MANGA-YOLO, MammoClean, MoRFSE, MV-Swin-T. 이 논문들의 기법이 실제로 코드에 이식되지는 않았으며, 문제 정의 및 방향 설정을 위한 참고 자료로 조사되었습니다.

---

*이 README와 저장소 구성은 원본 프로젝트("4학년 1학기 설계")의 실제 코드를 기준으로 정리되었으며, 개인정보(학번/이메일/실명)와 원본 데이터·모델 가중치·행정 문서는 포함하지 않았습니다.*
