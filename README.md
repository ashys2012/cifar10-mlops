# CIFAR-10 Image Classifier — ResNet18 + ZenML + MLflow + FastAPI

An end-to-end image classification system for the CIFAR-10 dataset.

---

File Structure

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

### Why ResNet18?
ResNet18 pretrained on ImageNet gives a strong baseline immediately, and transfer learning is the correct production approach for a 10-class image dataset of this size. The final classification head is replaced with a new linear layer matching the 10 CIFAR-10 classes.

### Why ZenML for the pipeline?
ZenML gives each training run automatic artifact versioning, step-level caching, and a lineage graph, things you would build by hand otherwise. The data loading step is cached (CIFAR-10 is a fixed public dataset), so re-running the pipeline only re-executes training and evaluation. Each run links through to its MLflow experiment automatically.

### Why SQLite for MLflow?
The file-store backend (`mlruns/`) was deprecated in recent MLflow versions and no longer supports the Model Registry. SQLite is the correct local backend as it supports the full registry API.

### Why the Model Registry?
Hardcoding a run ID into the server is fragile — it breaks every time you retrain. The MLflow Model Registry decouples training from serving: the server always loads the model aliased `champion`, and promoting a new model is a one-line registry operation, not a code change.

---

## Setup

### Prerequisites
- Python 3.12
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

Interactive docs: **http://localhost:8000/docs**

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

ResNet18 with a softmax head always produces a probability distribution that sums to 1.0, even for completely unrelated inputs (a photo of furniture, a medical scan, etc.). The model is forced to pick one of its 10 classes regardless of how dissimilar the input is to anything it was trained on; it has no concept of "I don't recognise this". For example, I added an X-ray image to the api and it predicted with 59% confidence that it is an automobile. 
To mitigate this issue, we can add a threshold (for example 60%) and if the confidence threshold is below 60%, then it should be classed as unknown.

**Alternate approaches:**
- **A dedicated "unknown" class** — collect a diverse set of non-CIFAR-10 images, add an 11th class, and fine-tune. The model then has an explicit bucket for out-of-distribution inputs.
---

### 2. Preparing for scale — batching and queuing

1. To address the performance bottlenecks of the current synchronous inference server, we can optimise the architecture by focusing on asynchronous task handling, throughput optimisation, and infrastructure scalability
The API accepts the request and immediately returns a job_id. The inference task is pushed to a background worker pool and once completed, notifies the user.

2. If the inference is deployed on a Kubernetes cluster, we can add horizontal scaling (for example, we can use Ray Serve). Ray serve can dynamically scale the number of inference replicas based on the current traffic, ensuring optimal resource utilisation and availability across the cluster. 
---

### 3. Feedback loop for unknown classes

If the endpoint receives a growing volume of low-confidence or rejected predictions, that signal should feed back into the model rather than being silently discarded.

To improve model accuracy over time, we will implement a "human-in-the-loop" workflow to handle edge cases. Any predictions where the model has low confidence will be automatically flagged and stored in a review database. A human expert will then inspect these samples to provide the correct label or tag them as "unknown" if they fall outside the standard CIFAR-10 classes. Once a sufficient number of these reviewed samples have been collected, the system will automatically trigger a pipeline to retrain the model with the new data, ensuring it continuously learns and evolves to handle difficult scenarios.

---

### 4. Detecting drift and performance degradation

We will implement proactive monitoring for prediction and confidence drift to detect when the model’s environment changes. By logging the distribution of predicted classes and the average softmax confidence scores, we can identify anomalies—such as a sudden surge in a specific class or a downward trend in certainty—that signal the input data is shifting away from the training distribution. We will use statistical methods to quantify these shifts, allowing us to distinguish between normal fluctuations and genuine model degradation.

---

### 5. Adversarial robustness

A standard ResNet trained on clean images is brittle to adversarial examples — imperceptible perturbations to pixel values that cause confident wrong predictions. This is a known property of softmax classifiers, not a bug specific to this implementation.

To ensure our model remains secure against malicious inputs, we must address its vulnerability to "adversarial examples"—images with tiny, invisible pixel changes designed to trick the model into making confident errors. To mitigate these risks, we can employ techniques like adversarial training, where the model learns to ignore these perturbations, or simple input preprocessing like smoothing to filter out noise. Ultimately, we treat these adversarial attacks as a form of "out-of-distribution" data, ensuring that any input the model wasn't trained to handle is either flagged or filtered to maintain system integrity.

---

## Further work (given more time)

- **Containerisation** — Dockerfile + docker-compose so the server and MLflow UI run together with a single `docker compose up`
- **Prometheus + Grafana** — structured metrics for prediction distribution, latency percentiles, and error rates
- **Hyperparameter Tuning** - By using the optuna or the hyperopts library
