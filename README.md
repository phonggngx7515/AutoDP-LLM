# AutoDP-LLM: Multiclass Intrusion Detection

A multiclass network intrusion detection pipeline built around LLM-generated
preprocessing and feature selection. Generate new artifacts through an API or
reuse existing artifacts, then evaluate them on a fixed TRAIN/VAL/TEST protocol.

## Repository structure

```text
.
├── experiments/
│   ├── run.py                  # Entry point
│   ├── config.py               # Multiclass LLM configuration and validation
│   ├── pipeline.py             # Artifact loading and orchestration helpers
│   └── validation_suite.py     # Resampling selection, TEST evaluation and reports
├── data/preprocessing/
│   ├── prior.py                # Dataset cleanup and multiclass target preparation
│   └── llm.py                  # Apply generated preprocessing
├── llm/
│   ├── api_caller.py           # Provider clients and usage/cost tracking
│   ├── llm_suggestions.py      # Three-stage code generation
│   ├── prompt_builders.py      # Stage prompts and contracts
│   └── validation.py          # Static and isolated runtime artifact checks
├── models/base_model.py        # Classifier factories and weighted multiclass metrics
├── tests/                      # Multiclass pipeline regression checks
├── datasets/                   # Official TRAIN/TEST CSV files; ignored by Git
├── generated/                  # Generated Python modules and history.json
├── rejected_artifacts/         # Generation validation diagnostics
├── text_files/                 # Reports and validation_splits/ manifests
├── .env.example                # Configuration template
├── .env                        # Local runtime settings; ignored by Git
└── requirements.txt            # Direct runtime dependencies
```

Output directories are created when needed. Datasets are not bundled with this
repository. Place official pre-split TRAIN/TEST CSVs in `datasets/` or configure
paths to existing files. Supported datasets are UNSW-NB15, NSL-KDD, and CIC-IDS-2017.

## Pipeline

1. Load official TRAIN and create/reuse a fixed split of approximately 80%
   inner TRAIN and 20% VAL with seed 42. Keep rare classes in inner TRAIN.
2. Apply dataset-specific cleanup and prepare the multiclass target.
3. Generate or load an artifact containing preprocessing and feature selection.
   New generation uses three stages: select a resampling family, implement
   preprocessing, and implement an adaptive feature selector. Generated code
   must pass static and isolated runtime validation before publication.
4. Select features once on the unresampled `none` candidate, then lock that
   feature list across resampling candidates. Use standard LightGBM on VAL to
   choose `none`, `mild`, or `moderate` when available. Within a weighted-F1
   difference strictly below 0.003, prefer the lighter resampling policy.
5. Lock each artifact's features and resampling policy, fit classifiers on
   inner TRAIN, and evaluate on official TEST. TEST is opened only after VAL
   selection; there is no full-TRAIN refit. Save reports and audit manifests.

The generated selector chooses K internally using semantic seeds, RF+MI evidence
at threshold 0.55, and Pearson redundancy filtering at 0.90. K is not clamped
or controlled by an environment variable. Two-category input features remain
valid predictors for the multiclass task.

Classifiers: Decision Tree, Random Forest, KNN, MLP, Bernoulli NB, XGBoost,
LightGBM, and CatBoost. Accuracy and weighted Precision/Recall/F1, timing,
and API usage are reported as means and standard deviations across successful artifacts.

## Setup and execution

Run all commands from the repository root. Initial setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
mkdir -p datasets
```

Edit `.env` with your dataset paths, provider, API key, and model ID:

```env
DATASET_NAME=UNSW_NB15
TRAIN_PATH=./datasets/UNSW_NB15_training-set.csv
TEST_PATH=./datasets/UNSW_NB15_testing-set.csv
CLASSIFICATION_TYPE=multiclass
TARGET_COLUMN=attack_cat
LLM_PROVIDER=openai
BASE_URL=https://openrouter.ai/api/v1
API_KEY=your-api-key
MODEL_ID=openai/gpt-5-codex
USE_LLM=true
N_RUNS=1
```

Start the pipeline:

```bash
.venv/bin/python experiments/run.py
```

To capture console output:

```bash
mkdir -p text_files
.venv/bin/python -u experiments/run.py > text_files/run.log 2>&1
```

For YesScale/Anthropic Messages, replace the provider settings with
`LLM_PROVIDER=yescale`, `BASE_URL=https://api.yescale.io`, and a supported
model ID, for example `MODEL_ID=claude-sonnet-4-6`. `LLM_MAX_TOKENS` defaults to 8192.

### Reuse generated artifacts without API calls

Keep the dataset settings and LLM model ID. For one existing file:

```env
USE_LLM=false
N_RUNS=1
GENERATED_FILE=artifact_01.py
```

For multiple artifacts, unset `GENERATED_FILE` and provide distinct filenames:

```env
USE_LLM=false
N_RUNS=3
GENERATED_FILES=artifact_01.py,artifact_02.py,artifact_03.py
```

Replace the example names with actual `.py` filenames in `generated/`. Artifacts
must implement the current resampling contract and the three-argument
`reduce_features(X_train, X_test, y_train)` API. Reuse still performs VAL/TEST
evaluation; it does not trigger code generation.

### Regression checks

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

## Configuration

`.env` values override shell environment variables.

| Variable | Meaning |
| --- | --- |
| `DATASET_NAME` | `UNSW_NB15` (default), `NSL_KDD`, or `CIC_IDS_2017`. |
| `TRAIN_PATH`, `TEST_PATH` | Required paths to official pre-split CSVs; relative paths resolve from the working directory. |
| `CLASSIFICATION_TYPE` | Only `multiclass` is accepted; this is also the default. |
| `TARGET_COLUMN` | Target after cleanup: `attack_cat` for UNSW-NB15/NSL-KDD, `Label` for CIC-IDS-2017. Defaults depend on the dataset; NSL-KDD normalizes the setting to `attack_cat`. |
| `MODEL_ID` | Required API model ID for the LLM that generates code. Evaluation uses the eight classifiers listed above. |
| `USE_LLM` | `true` generates fresh artifacts; `false` loads existing ones. Default: `true`. |
| `N_RUNS` | Number of artifacts to generate/reuse; integer >= 1, default: `1`. Generation seeds start at 42. |
| `GENERATED_FILE` | Python filename in `generated/` for one reused artifact. |
| `GENERATED_FILES` | Comma-separated filenames; exactly `N_RUNS` distinct names for multiple reused artifacts. Do not combine with `GENERATED_FILE`. |
| `LLM_PROVIDER` | `openai` / `openai-compatible` or `anthropic` / `yescale`; default: `openai`. |
| `BASE_URL`, `API_KEY` | Provider endpoint and key. Fallbacks: `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL` and `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`. |
| `LLM_MAX_TOKENS` | Maximum output tokens for Anthropic/YesScale; default: `8192`. |
| `PRICE_PROMPT_PER_1M_USD`, `PRICE_COMPLETION_PER_1M_USD` | USD per million tokens for cost reporting; default: `0`. Provider prices are not fetched automatically. |

For NSL-KDD, use `TRAIN_PATH=./datasets/KDDTrain+.csv` and
`TEST_PATH=./datasets/KDDTest+.csv`. Its raw attack labels are mapped to
`normal`, `DoS`, `Probe`, `R2L`, and `U2R` before stratified selection.

## Outputs

- `generated/*.py` and `generated/history.json`: generated modules and generation history.
- `text_files/*.txt` and `*.json`: reports and manifests with selected policies,
  locked features, weighted metrics, timing, API usage, and artifact failures.
- `text_files/validation_splits/*.json`: fixed split positions and dataset integrity metadata.
- `rejected_artifacts/`: diagnostics from generated responses that failed validation.

## Citation

```bibtex
@article{Nguyen_2026,
   title={AutoDP-LLM: automating data pre-processing for intrusion detection systems using large language models},
   volume={82},
   ISSN={1573-0484},
   url={http://dx.doi.org/10.1007/s11227-026-08894-8},
   DOI={10.1007/s11227-026-08894-8},
   number={15},
   journal={The Journal of Supercomputing},
   publisher={Springer Science and Business Media LLC},
   author={Nguyen, Bao-Phong and Pham, Gia-Khanh and Do, Thai-Duong and Trang, Mai Xuan and Le, Minh-Tuan and Tran, Xuan-Nam and Vu, Huan and Nguyen, Tien-Cuong and Ngo, Vu-Duc and Van Luong, Thien},
   year={2026},
   month=Oct }

```
