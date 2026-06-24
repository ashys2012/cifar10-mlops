# CIFAR-10 Image Classifier — ResNet18 + ZenML + MLflow + FastAPI

A production-style image classification system for the CIFAR-10 dataset. The goal was not just to train a model, but to engineer a clean, reproducible, and extensible ML system — with a tracked training pipeline, a model registry, and a served REST API.

---

## What's in here

```
cifar_10/
├── app.py                        # FastAPI inference server
├── components/
│   ├── model.py                  # ResNet18 with custom classifier head
│   ├── tracking.py               # MLflow helpers (git info, logging, reproduce cmd)
│   ├── common.py                 # Shared dataset class + transforms
│   ├── data_loader_step.py       # ZenML step: downloads CIFAR-10
│   ├── train_step.py             # ZenML step: training loop + MLflow logging
│   ├── evaluate_step.py          # ZenML step: test-set evaluation
│   ├── pipeline.py               # Wires the three steps into a ZenML pipeline
│   └── run.py                    # CLI entrypoint
├── register_best_run.py          # One-time script: promotes best run to registry
├── requirements.txt
└── mlflow.db                     # SQLite MLflow tracking store (local)
```

---

## Architecture decisions

### Why ResNet18 and not a custom CNN?
CIFAR-10 is a solved benchmark. The task explicitly says we are not being tested on whether we can train a CNN — we are being tested on **engineering quality**. ResNet18 pretrained on ImageNet gives a strong baseline immediately, and transfer learning is the correct production approach for a 10-class image dataset of this size. The final classification head is replaced with a new linear layer matching the 10 CIFAR-10 classes.

### Why ZenML for the pipeline?
ZenML gives each training run automatic artifact versioning, step-level caching, and a lineage graph — things you would build by hand otherwise. The data loading step is cached (CIFAR-10 is a fixed public dataset), so re-running the pipeline only re-executes training and evaluation. Each run links through to its MLflow experiment automatically.

### Why SQLite for MLflow?
The file-store backend (`mlruns/`) was deprecated in recent MLflow versions and no longer supports the Model Registry. SQLite is the correct local backend — it supports the full registry API (register, alias, stage promotion) with no extra infrastructure.

### Why the Model Registry?
Hardcoding a run ID into the server is fragile — it breaks every time you retrain. The MLflow Model Registry decouples training from serving: the server always loads the model aliased `champion`, and promoting a new model is a one-line registry operation, not a code change.

---

## Setup

### Prerequisites
- Python 3.12 (3.13 has known compatibility issues with some dependencies — stick to 3.12)
- [`uv`](https://github.com/astral-sh/uv) for environment and dependency management
- A GPU is optional — the server and training loop both fall back to CPU

### Install dependencies

```bash
# uv creates and manages the virtual environment automatically
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# ZenML integrations — installs MLflow + PyTorch with tested versions
zenml init
zenml integration install mlflow pytorch -y
```

### Register the ZenML stack

```bash
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="sqlite:////absolute/path/to/cifar_10/mlflow.db"

zenml stack register cv_stack \
    -e mlflow_tracker \
    -o default \
    -a default

zenml stack set cv_stack
```

> **Why `--tracking_uri` here?** This tells ZenML which MLflow store to write to. Using SQLite (not the default file store) is required for the Model Registry to work.

---

## Running the training pipeline

```bash
python components/run.py
# with options:
python components/run.py --epochs 15 --batch-size 128 --lr 0.0005
```

This runs three ZenML steps in sequence:

1. **`data_loader_step`** — downloads CIFAR-10 via torchvision, returns raw uint8 arrays. Cached after first run.
2. **`train_step`** — fine-tunes ResNet18, logs loss/accuracy per epoch, logs the model artifact to MLflow.
3. **`evaluate_step`** — runs the trained model on the held-out test set, logs final metrics to MLflow.

### Viewing results

```bash
# Option A — ZenML dashboard (pipeline lineage, artifact viewer, MLflow links)
zenml login --local

# Option B — MLflow UI directly
mlflow ui --backend-store-uri sqlite:////absolute/path/to/cifar_10/mlflow.db
# then open http://127.0.0.1:5000
```

---

## Promoting a model to the registry

After training, run this once to register the best run and tag it as `champion`:

```bash
python register_best_run.py
```

This script:
1. Finds the finished run with the highest `val_accuracy` automatically
2. Registers it in the MLflow Model Registry under `cifar10-resnet18`
3. Sets the `champion` alias so the server knows which version to load

To promote a new model after retraining, run the same script again — it will register the new best run as a new version and move the `champion` alias.

---

## Running the inference server

```bash
python app.py
# or:
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server loads the model at startup using this priority order:

| Priority | Source | When it applies |
|---|---|---|
| 1 | `MODEL_URI` env var | Explicit override for debugging |
| 2a | Registry `champion` alias | After `register_best_run.py` has been run |
| 2b | Latest registered version | If alias not set yet |
| 3 | Best finished run by `val_accuracy` | Before any model is registered |

This means the server works immediately after training, even before you've touched the registry.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, returns device info |
| `GET` | `/metadata` | Model architecture, class names, loaded URI |
| `GET` | `/class_names` | List of the 10 class labels |
| `POST` | `/predict` | Upload an image → returns top predicted class |
| `POST` | `/predict_with_confidence` | Upload an image → returns top-k classes with confidence scores |

Interactive docs (try it in the browser): **http://localhost:8000/docs**

### Example request

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/image.jpg"
# {"predicted_class": "dog"}

curl -X POST "http://localhost:8000/predict_with_confidence?top_k=3" \
     -F "file=@/path/to/image.jpg"
# {"predictions": [{"class": "dog", "confidence": 0.874}, ...]}
```

---

## Environment variables

All defaults work out of the box. Override any of these to customise behaviour without touching code:

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Path to local `mlflow.db` | Where MLflow reads/writes runs |
| `MODEL_URI` | *(not set)* | Skip registry lookup, load this URI directly |
| `MODEL_NAME` | `cifar10-resnet18` | Registry model name |
| `MLFLOW_EXPERIMENT` | `training_pipeline` | Experiment to search for best run |
| `ACCURACY_METRIC` | `val_accuracy` | Metric used to rank runs in fallback mode |

---

## Design notes

### Raw arrays, not pre-transformed tensors
`data_loader_step` passes raw `(N, 32, 32, 3)` uint8 arrays between steps, not pre-resized float tensors. Resizing all 50,000 training images to 224×224 and storing them as float32 would require ~30 GB of RAM. The `ArrayImageDataset` in `common.py` applies the resize/normalise/augment transform per sample inside the DataLoader — identical behaviour, deferred until each batch is pulled.

### Step caching
`data_loader_step` has `enable_cache=True`. CIFAR-10 is a fixed public dataset, so re-running with the same `data_dir` is safe to skip. If you swap in a versioned dataset later (e.g. tracked with DVC), set `enable_cache=False` — ZenML's cache key is the `data_dir` string, not file content, so it won't detect changes automatically.

### Single worker on GPU
The server runs with `workers=1`. A PyTorch model loaded on a GPU lives on that one device — multiple workers would each try to allocate their own copy, which breaks on single-GPU machines. For CPU-only deployments, `workers=4` is safe.

---

## Testing

Tests are fully offline — no MLflow tracking server or trained model is needed. The model is replaced with a mock that always returns a fixed logit vector.

```bash
uv pip install pytest httpx
pytest tests/ -v
```

The test suite covers:

- `/health`, `/metadata`, `/class_names` — response shape and status codes
- `/predict` — valid PNG/JPEG, wrong extension (415), oversized file (413), corrupt bytes (400), missing file (422)
- `/predict_with_confidence` — correct count for `top_k`, confidences sum to 1.0, sorted descending, out-of-range `top_k` (400)
- Root `/` returns 404 as expected (no route defined)

---

## Research questions

### 1. Out-of-distribution images — how does the model handle them, and how could it indicate "unknown"?

ResNet18 with a softmax head always produces a probability distribution that sums to 1.0, even for completely unrelated inputs (a photo of furniture, a medical scan, pure noise). The model is forced to pick one of its 10 classes regardless of how dissimilar the input is to anything it was trained on — it has no concept of "I don't recognise this".

**Current partial mitigation:** the `/predict_with_confidence` endpoint exposes the top-k softmax scores. A caller can apply a confidence threshold (e.g. reject if `max(softmax) < 0.6`) as a simple heuristic — genuinely uncertain inputs tend to produce flatter distributions.

**Better approaches:**
- **Temperature scaling** — a single learned scalar `T` applied before softmax that makes the model better-calibrated without retraining. Confidence scores become more meaningful as a proxy for correctness.
- **Energy-based OOD detection** — compute `E(x) = -log Σ exp(logit_i)`. In-distribution inputs cluster at lower energy; OOD inputs have measurably higher energy. This can be thresholded without any retraining.
- **A dedicated "unknown" class** — collect a diverse set of non-CIFAR-10 images, add an 11th class, and fine-tune. The model then has an explicit bucket for out-of-distribution inputs.
- **`/predict` returning a structured rejection** — rather than always returning a class, add a `rejected: true` field when confidence is below threshold, so callers handle it explicitly rather than silently trusting a bad prediction.

---

### 2. Preparing for scale — batching and queuing

The current server processes one image per request synchronously. Under high load this has two problems: requests queue at the HTTP layer, and the GPU is underutilised (most of each forward pass is spent on overhead rather than compute, because batch size is 1).

**Request queuing:** put the server behind a message queue (Redis + Celery, or AWS SQS). Clients POST an image and receive a job ID immediately; a worker picks it up, runs inference, and writes the result. Clients poll `/result/{job_id}` or receive a webhook. This decouples client latency from inference latency entirely.

**Dynamic batching:** collect incoming requests over a short window (e.g. 20ms) and run a single forward pass on the batch. A batch of 32 images takes only slightly longer than a batch of 1 on a GPU, so throughput increases dramatically. Libraries like NVIDIA Triton Inference Server implement this natively. A simpler DIY version: a background thread drains a queue, accumulates images until the batch is full or the timeout fires, runs inference, and maps results back to waiting futures.

**Horizontal scaling:** run multiple server replicas behind a load balancer (nginx, or a Kubernetes Service). Each replica loads its own copy of the model. For GPU deployments, one replica per GPU is the right unit.

---

### 3. Feedback loop for unknown classes

If the endpoint receives a growing volume of low-confidence or rejected predictions, that signal should feed back into the model rather than being silently discarded.

**Proposed pipeline:**
1. Log every rejected prediction (image hash, timestamp, top confidence score) to a database table.
2. Build a lightweight review queue — a human labels a sample of flagged images (or a secondary, higher-capacity model does a first pass).
3. Once enough labelled examples accumulate, trigger a retraining run via the existing ZenML pipeline, now with the new examples included in the training set.
4. The retrained model goes through the normal registration flow — `register_best_run.py` promotes it to `champion` only if it beats the current champion's validation accuracy.

This is **active learning**: the production system surfaces its own blind spots, and the feedback loop closes them systematically rather than waiting for someone to notice a problem.

---

### 4. Detecting drift and performance degradation

**Prediction distribution drift:** log the predicted class for every request. In steady state, the distribution of predictions should be roughly stable. If "ship" suddenly accounts for 40% of predictions when it was historically 10%, that's a signal — either the input distribution has shifted, or the model has degraded. Population Stability Index (PSI) is the standard metric for this.

**Confidence drift:** log the max softmax score per request. A falling average confidence over time (without a corresponding rise in explicit rejections) suggests the model is becoming less certain — possibly because inputs are drifting away from the training distribution.

**Latency and error rates:** expose a `/metrics` endpoint (Prometheus format) tracking request count, error count by status code, and inference latency percentiles (p50, p95, p99). Wire this into Grafana with alerts on error rate > 1% or p99 latency > 500ms.

**Ground truth comparison (where possible):** if a downstream system eventually produces a label for a prediction (e.g. a user corrects a wrong classification), log prediction vs. ground truth. Track rolling accuracy over a sliding window and alert if it drops below a threshold.

---

### 5. Adversarial robustness

A standard ResNet trained on clean images is brittle to adversarial examples — imperceptible perturbations to pixel values that cause confident wrong predictions. This is a known property of softmax classifiers, not a bug specific to this implementation.

**Demonstrating the vulnerability:**
```python
# pip install torchattacks
import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
adv_image = atk(image_tensor, true_label)
# model(adv_image) will likely predict the wrong class with high confidence
```
PGD (Projected Gradient Descent) is the standard benchmark attack. Measuring accuracy on PGD-attacked CIFAR-10 test images gives a meaningful robustness score.

**Defences:**
- **Adversarial training** — include adversarial examples in the training set (generated on-the-fly during each epoch). This is the most effective known defence but increases training time by 3–5×.
- **Input preprocessing** — JPEG compression, Gaussian smoothing, or feature squeezing before the forward pass destroys many adversarial perturbations at low cost.
- **Certified defences** — randomised smoothing (Cohen et al. 2019) provides a provable robustness guarantee: for any input within an L2 ball of radius r, the prediction is certified correct. This comes at an accuracy cost on clean inputs.

In a production system, adversarial inputs are best treated as a subset of the OOD problem — both are inputs the model was not designed for, and the same confidence-thresholding and logging infrastructure catches them.

---

## Further work (given more time)

- **Containerisation** — Dockerfile + docker-compose so the server and MLflow UI run together with a single `docker compose up`
- **CI pipeline** — GitHub Actions: lint → unit tests → 1-epoch smoke-test run → register if accuracy threshold met
- **Grad-CAM `/explain` endpoint** — return a heatmap image overlaid on the input showing which pixels drove the prediction (addresses explainability)
- **Async model hot-swap** — background thread polls the registry every N minutes; if a new `champion` is found, reload the model without restarting the server
- **Prometheus + Grafana** — structured metrics for prediction distribution, latency percentiles, and error rates
