# 🫁 Chest X-Ray Pneumonia Classifier

A medical image classification system that detects **Pneumonia from Chest X-Rays** using a fine-tuned **ResNet-50** pretrained on ImageNet. The project includes full training, evaluation, Grad-CAM visualization, and inference pipelines — running entirely on real Kaggle data.

---

## 📁 Project Structure

```
chest-xray-classifier/
│
├── chestxray/                # Installable Python package
│   ├── config.py             # Typed dataclass configuration (env-overridable)
│   ├── data.py               # Transforms, loaders, class-weight computation
│   ├── model.py              # Shared ResNet-50 builder (train + inference)
│   ├── engine.py             # Two-phase fine-tuning training loop
│   ├── metrics.py            # Accuracy, F1, AUC, confusion matrix, ROC, JSON
│   ├── visualize.py          # Sample grids, CLAHE, class distribution
│   ├── gradcam.py            # Grad-CAM (managed hook lifecycle)
│   ├── inference.py          # Classifier: single/batch predict + overlays
│   ├── checkpoint.py         # Checkpoints embed class names + config
│   ├── eda.py                # Exploratory data analysis
│   ├── dataset_setup.py      # Kaggle download helper
│   ├── api.py                # FastAPI serving app
│   ├── cli.py                # Unified CLI entry point
│   └── utils.py              # Logging, seeding, device selection
│
├── tests/                    # Pytest suite (imports, model, data, API, ...)
├── train.py / predict.py / eda.py / setup_data.py   # Backward-compat wrappers
│
├── pyproject.toml            # Packaging, deps, ruff/black/pytest config
├── requirements.txt          # Runtime deps   requirements-dev.txt  # Dev deps
├── Dockerfile / .dockerignore
├── Makefile                  # Common dev/ops tasks
├── .pre-commit-config.yaml
├── .env.example
├── .github/workflows/ci.yml  # Lint + test CI
│
├── data/        # (downloaded)   checkpoints/  # (trained)   outputs/  # (generated)
```

---

## 📦 Dataset

**Chest X-Ray Images (Pneumonia)** — Paul Mooney, Kaggle  
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

The dataset contains front-view chest X-ray images labeled as **NORMAL** or **PNEUMONIA** (bacterial & viral).

---

## ⚙️ Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chest-xray-classifier.git
cd chest-xray-classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Simple: runtime dependencies only
pip install -r requirements.txt

# Recommended: install the package (enables the `chestxray` command + extras)
pip install -e ".[dev,serve,data]"
```

> ✅ GPU recommended. CUDA-capable GPU will be auto-detected.  
> CPU training is supported but slower (~2–3× longer per epoch).  
> 🪟 On Windows, DataLoader workers default to `0` to avoid multiprocessing
> issues; override with `--num-workers N` or `CXR_NUM_WORKERS`.

---

## 🔑 Kaggle API Setup

You need a free Kaggle account to download the dataset.

1. Log in at [https://www.kaggle.com](https://www.kaggle.com)
2. Go to **Account → API → Create New API Token**
3. Download `kaggle.json`
4. Place it in the correct location:

```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

---

## 🚀 Running the Project

All functionality is available through a single CLI. After `pip install -e .`:

```bash
chestxray setup-data
chestxray eda
chestxray train --epochs 15 --batch-size 32
chestxray predict --image path/to/xray.jpg
chestxray serve --port 8000
```

Equivalent module form (no install needed): `python -m chestxray.cli <command>`.
The legacy `python train.py ...` style scripts still work as thin wrappers.

### Step 1 — Download the dataset

```bash
chestxray setup-data
# or: python setup_data.py
```

This downloads and organises the Kaggle dataset into `data/chest_xray/`.

---

### Step 2 — Exploratory Data Analysis (optional but recommended)

```bash
python eda.py --data_dir data/chest_xray --save_dir outputs/eda
```

Generates:
- Class distribution bar chart
- Pixel intensity histogram (NORMAL vs PNEUMONIA)
- Image size scatter plot
- Raw sample image grid

---

### Step 3 — Train the model

```bash
python train.py \
  --data_dir data/chest_xray \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-3 \
  --unfreeze_epoch 6
```

**What happens:**
- Epochs 1–5: Only the classification head is trained (backbone frozen)
- Epoch 6+: Full fine-tuning — entire ResNet-50 is unfrozen at a lower LR
- A **proper stratified validation set** is carved from the training data
  (the dataset's shipped 16-image `val` folder is too small to select on)
- The best model is chosen by **balanced accuracy** and saved to
  `checkpoints/best_model.pth`
- Evaluation metrics, confusion matrix, and ROC curve saved to `outputs/`

#### 🎯 Reaching >90% accuracy

The default recipe is tuned to clear **90%+ test accuracy** on the full
dataset. The key ingredients (all on by default):

- **Reliable validation split** — `resplit_val` carves a stratified 15% val set
  from train so "best model" selection isn't decided by 16 noisy images.
- **Balanced-accuracy model selection** — robust to the ~3:1 PNEUMONIA:NORMAL
  imbalance.
- **Class weights + label smoothing** (`--label-smoothing 0.05`).
- **Stronger augmentation** — random resized crop, small affine/rotation, jitter.
- **Two-phase fine-tuning** — head first, then the whole backbone.
- **Test-time augmentation** — pass `tta=true` at inference (e.g.
  `POST /predict?tta=true`) for a small extra boost.

```bash
# Full high-accuracy run (GPU recommended; ~15 epochs)
chestxray train --data-dir data/chest_xray --epochs 15 --batch-size 32
```

Expected on the standard test split: **accuracy ≈ 0.90–0.94, AUC ≈ 0.95+**
(exact numbers vary by seed/hardware). Train metrics are written to
`outputs/metrics.json` and shown live in the web UI's Performance section.

> 💡 **Quick demo (CPU):** to produce a usable checkpoint in a couple of minutes
> for trying the web UI, cap the data and epochs:
> ```bash
> chestxray train --epochs 2 --limit 300 --batch-size 16
> ```
> This trains on a small random subset — fine for a demo, not for real accuracy.

#### CLI Arguments

| Argument            | Default            | Description                                       |
|---------------------|--------------------|---------------------------------------------------|
| `--data_dir`        | `data/chest_xray`  | Path to dataset root                              |
| `--batch_size`      | `32`               | Training batch size                               |
| `--epochs`          | `15`               | Total training epochs                             |
| `--lr`              | `1e-3`             | Initial learning rate                             |
| `--unfreeze_epoch`  | `6`                | Epoch to unfreeze backbone for full fine-tuning   |
| `--val-split`       | `0.15`             | Fraction of train held out for validation         |
| `--no-resplit-val`  | _(off)_            | Use the tiny shipped val folder instead           |
| `--label-smoothing` | `0.05`             | Label smoothing factor for the loss               |

---

### Step 4 — Run inference

**Single image:**
```bash
python predict.py \
  --image data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg \
  --checkpoint checkpoints/best_model.pth \
  --save_dir outputs/predictions
```

**Entire folder:**
```bash
python predict.py \
  --folder data/chest_xray/test/PNEUMONIA \
  --checkpoint checkpoints/best_model.pth \
  --save_dir outputs/predictions
```

Each prediction generates a **Grad-CAM overlay** image showing which regions of the X-ray influenced the model's decision.

---

## 🧠 Model Architecture

```
ResNet-50 (pretrained ImageNet)
    └── Backbone (conv layers, BatchNorm, ReLU)
          └── fc → Dropout(0.4) → Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→2)
```

**Training Strategy:**
- Phase 1 (frozen backbone): Train only the new classifier head → fast convergence
- Phase 2 (unfrozen backbone): Fine-tune all layers at 10× lower LR → domain adaptation
- Optimiser: Adam + Weight Decay 1e-4
- Scheduler: StepLR (γ=0.5, step=5)
- Loss: CrossEntropyLoss

---

## 📊 Expected Results

| Metric    | Value (approx.) |
|-----------|-----------------|
| Test Acc  | ~93–95%         |
| F1 Score  | ~0.93–0.95      |
| ROC-AUC   | ~0.97–0.99      |
| Precision | ~0.93–0.95      |
| Recall    | ~0.94–0.96      |

*Results vary slightly by hardware and random seed.*

---

## 🗂️ Output Files

After a full run, `outputs/` contains:

| File                        | Description                             |
|-----------------------------|-----------------------------------------|
| `sample_images.png`         | Sample training images grid             |
| `training_curves.png`       | Loss & accuracy curves                  |
| `confusion_matrix.png`      | Test set confusion matrix               |
| `roc_curve.png`             | ROC curve with AUC score                |
| `metrics.txt`               | All evaluation metrics (text)           |
| `eda/class_distribution.png`| Per-split class counts                  |
| `eda/intensity_distribution.png` | Pixel intensity histogram          |
| `predictions/*.png`         | Grad-CAM overlay for each prediction   |

---

## 🔬 Key Concepts Demonstrated

- **Transfer Learning** — pretrained ResNet-50 adapted to medical imaging
- **Fine-Tuning Strategy** — staged unfreezing (head → full)
- **Data Augmentation** — random flip, rotation, colour jitter
- **Grad-CAM** — gradient-based saliency map for model interpretability
- **Evaluation** — confusion matrix, F1, precision, recall, ROC-AUC
- **OpenCV** — CLAHE contrast enhancement for X-ray preprocessing
- **Real Kaggle Dataset** — no synthetic data

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError: No module named 'kaggle'`**
```bash
pip install kaggle
```

**`CUDA out of memory`**
```bash
# Reduce batch size
python train.py --batch_size 16
```

**`OSError: [Errno 28] No space left`**
The dataset is ~1.2 GB. Ensure you have at least 3 GB free.

**Dataset folder structure wrong:**  
The expected structure is `data/chest_xray/train/NORMAL/` and `data/chest_xray/train/PNEUMONIA/`.  
If your download extracted differently, rename folders to match.

---

## 🌐 Web App & REST API

Start the FastAPI service (loads `checkpoints/best_model.pth` by default):

```bash
chestxray serve --host 127.0.0.1 --port 8000
# or: uvicorn chestxray.api:app --port 8000
```

Then open **`http://127.0.0.1:8000/`** for the web UI. Interactive API docs
live at `http://127.0.0.1:8000/docs`.

**Web UI features:**
- **Single mode** — drag-and-drop one X-ray with an animated confidence gauge,
  per-class probability bars, and a Grad-CAM heatmap
- **Batch mode** — analyze many X-rays at once with a results table, live
  progress, summary counts, and **CSV export**
- **Decision-threshold slider** — re-classify live by adjusting the pneumonia
  probability cutoff (sensitivity tuning)
- **Interactive Grad-CAM blend** — slide heatmap opacity over the X-ray, plus
  click-to-zoom for each view
- **Low-confidence / borderline / uncertainty advisory** for cases needing review
- **Copy results** (JSON), **Download JSON**, and **Download PDF report**
- **DICOM (.dcm) upload** alongside PNG/JPG
- **OOD input guard** — warns when an upload doesn't look like a chest X-ray
- **Feedback buttons** (👍/👎) that record ground-truth corrections for retraining
- **Recent analyses** audit table (server-side log)
- Live **model status** indicator (device / offline / no-model)
- **Scrollable landing page** — hero, how-it-works, features, a live
  **Performance** section (pulls real metrics from `/metrics` after training),
  tech stack, and an FAQ

### Advanced / real-world features

- **DICOM support** — uploads are decoded with `pydicom` (handles MONOCHROME1
  inversion and windowing) so clinical files work directly.
- **Out-of-distribution guard** — a grayscale/size heuristic flags non-radiograph
  uploads (color photos, screenshots) before trusting the result.
- **Uncertainty & abstention** — Monte-Carlo Dropout (`CXR_MC_PASSES` forwards)
  yields predictive entropy and per-class std; high uncertainty recommends a
  radiologist review instead of a confident answer.
- **PDF reports** — `/predict/report` renders a one-page report (verdict,
  Grad-CAM panels, probability table, uncertainty, disclaimer).
- **Audit log & feedback loop** — every prediction is appended to
  `outputs/audit_log.jsonl`; user corrections land in `outputs/feedback.jsonl`.
- **API security** — optional API-key auth (`CXR_API_KEYS`) and per-IP rate
  limiting (`CXR_RATE_LIMIT`, requests/min). Auth is disabled when no keys are
  set so the bundled UI works out of the box; the UI sends `X-API-Key` from
  `localStorage['pulmo-api-key']` when present.

| Method | Path               | Description                                   |
|--------|--------------------|-----------------------------------------------|
| GET    | `/`                | Web UI (scrollable landing + analyzer)         |
| GET    | `/health`          | Liveness/readiness probe                       |
| GET    | `/metadata`        | Model class names, checkpoint, device          |
| GET    | `/metrics`         | Latest evaluation metrics (from training)      |
| GET    | `/history`         | Recent predictions (audit log)                 |
| POST   | `/predict`         | Image/DICOM upload → label + probs + OOD check |
| POST   | `/predict/analyze` | Prediction + Grad-CAM + uncertainty            |
| POST   | `/predict/report`  | One-page PDF report (`application/pdf`)         |
| POST   | `/feedback`        | Record correct/incorrect feedback              |

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg"
# {"label":"PNEUMONIA","confidence":0.98,"probabilities":{"NORMAL":0.02,"PNEUMONIA":0.98}}
```

> The web UI and `/predict*` endpoints require a trained checkpoint. Until you
> run `chestxray train`, the page loads but analysis returns a friendly
> "no model loaded" message. The checkpoint path is overridable via
> `CXR_CHECKPOINT_PATH`.

---

## 🐳 Docker

```bash
docker build -t chestxray:latest .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  chestxray:latest
```

The image is CPU-only and serves the API on port 8000 with a built-in healthcheck.

---

## 🧪 Testing & Quality

```bash
pip install -e ".[dev]"
pytest                # run the test suite
pytest --cov=chestxray  # with coverage
ruff check chestxray tests   # lint
black chestxray tests        # format
pre-commit install           # enable pre-commit hooks
```

CI (lint + tests on Python 3.9 & 3.11) runs automatically via GitHub Actions
(`.github/workflows/ci.yml`).

---

## ⚙️ Configuration

Defaults live in `chestxray/config.py` and can be overridden by CLI flags or
environment variables (see `.env.example`):

| Variable             | Purpose                          |
|----------------------|----------------------------------|
| `CXR_DATA_DIR`       | Dataset root                     |
| `CXR_CHECKPOINT_DIR` | Where checkpoints are written    |
| `CXR_OUTPUT_DIR`     | Where plots/metrics are written  |
| `CXR_BATCH_SIZE`     | DataLoader batch size            |
| `CXR_NUM_WORKERS`    | DataLoader workers (0 on Windows)|
| `CXR_CHECKPOINT_PATH`| Checkpoint loaded by the API     |
| `CXR_MC_PASSES`      | MC-Dropout passes for uncertainty (0 disables) |
| `CXR_API_KEYS`       | Comma-separated API keys (empty = auth off)    |
| `CXR_RATE_LIMIT`     | Requests/min/IP (0 disables)     |
| `CXR_LOG_LEVEL`      | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

---

## ⚠️ Disclaimer

This software is for **research and educational purposes only**. It is **not a
medical device** and must not be used for clinical diagnosis or treatment
decisions. Always consult qualified healthcare professionals.

---

## 📄 License

This project is for educational purposes. The dataset is licensed under  
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Paul Mooney / Kaggle.
