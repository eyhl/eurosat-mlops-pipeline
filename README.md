# EuroSAT MLOps Pipeline
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/tests.yml/badge.svg)
![example workflow](https://github.com/eyhl/group5-pyg-dtu-mlops/actions/workflows/coverage.yml/badge.svg)  

Train → evaluate → predict for EuroSAT RGB (10-class land use classification), with
reproducible splits and traceable run artifacts.

This repository exists as a _minimal but professional_ MLOps portfolio project:
- Reproducibility: deterministic splits + seeds
- Traceability: every run saves config + git state + metrics + model
- Robust CLI: config-driven training and run-folder based evaluation/inference

## Workflow

1) **Data**: EuroSAT RGB organized as ImageFolder (class subfolders)
2) **Train**: fine-tune ResNet18 head (default)
3) **Evaluate**: compute test metrics + confusion matrix
4) **Predict**: run inference on a folder of images and output `predictions.csv`

## Setup

Requirements: Python 3.11 (CI uses 3.11; local can also work on 3.10+).

Using `uv`:

```bash
uv venv
uv pip install -e ".[dev]"
```

Or using the Makefile:

```bash
make setup
```

## Dataset

Expected layout (ImageFolder-style):

```
data/eurosat_rgb/
  AnnualCrop/
    img1.jpg
  Forest/
    img2.jpg
  ... (10 classes)
```

Notes:
- The dataset is not downloaded automatically.
- Put the EuroSAT RGB dataset under `data/eurosat_rgb/`.

## Run

Train:

```bash
python -m src.train --config configs/default.yaml
# or
make train
```

Evaluation (use the `run_id` printed by training):

```bash
python -m src.evaluate --run artifacts/<run_id>
# or
make eval RUN=artifacts/<run_id>
```

Predict:

```bash
python -m src.predict --run artifacts/<run_id> --input path/to/images --output predictions.csv
# or
make predict RUN=artifacts/<run_id> INPUT=path/to/images OUTPUT=predictions.csv
```

## Artifacts

Each training run creates `artifacts/<run_id>/` containing:

- `model.pt`: best checkpoint (by validation accuracy)
- `metrics.json`: training curves + best epoch + test metrics (after evaluation)
- `config_used.yaml`: exact config snapshot used for the run
- `git_info.json`: commit hash + dirty flag (when available)
- `split.json`: deterministic train/val/test indices
- `class_to_idx.json`: class mapping used by `ImageFolder`
- `confusion_matrix.png`: saved by `src.evaluate` (if enabled)

## Reproducibility notes

- Seeds are set for `random`, `numpy`, and `torch`.
- Train/val/test split is deterministic (seeded) and persisted in `split.json`.
- Run IDs use `UTC timestamp + short git hash` when available.

## Device selection

The code automatically selects:
- `cuda` if available
- else `mps` (Apple Silicon)
- else `cpu`

The chosen device is logged at startup in each CLI.

## Limitations

- This is a small portfolio pipeline (not production infrastructure).
- No automatic dataset download (by design; reduces hidden network variability).

## Development

Run lint + tests:

```bash
ruff check .
pytest
```

Run a single test:

```bash
pytest tests/test_smoke.py::test_forward_pass
```
