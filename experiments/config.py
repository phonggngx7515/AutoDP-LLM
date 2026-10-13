"""
Experiment configuration — reads all settings from environment variables.

Usage:
    from experiments.config import load_config
    config = load_config()
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

VALID_DATASETS = {"UNSW_NB15", "CIC_IDS_2017", "NSL_KDD", "BotNetIoT_L01"}
VALID_CLASSIFICATION_TYPES = {"binary", "multiclass"}


@dataclass
class ExperimentConfig:
    dataset_name: str
    train_path: str
    test_path: str
    classification_type: str        # "binary" | "multiclass"
    model_id: str                   # "traditional" or API model ID (e.g. "gpt-4o", "openai/gpt-4.1")
    target_column: str              # The label column name in the dataset (e.g. "label", "class", "attack_cat")
    reduction_method: str = "selection"  # "selection" | "extraction" | "ae_extraction"
    k_values: list = field(default_factory=list)
    use_llm: bool = True
    generated_file: str | None = None     # Existing generated/ filename; used when use_llm is false
    generated_files: list[str] = field(default_factory=list)
    n_runs: int = 1
    resampling_selection_k: int = 28

    @property
    def llm_model(self) -> str:
        return self.model_id


def load_config() -> ExperimentConfig:
    load_dotenv(override=True)

    dataset_name = os.environ.get("DATASET_NAME", "UNSW_NB15")
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"DATASET_NAME must be one of {VALID_DATASETS}, got '{dataset_name}'")

    train_path = os.environ.get("TRAIN_PATH")
    test_path = os.environ.get("TEST_PATH")
    if not train_path or not test_path:
        raise ValueError("TRAIN_PATH and TEST_PATH must be set in .env")

    classification_type = os.environ.get("CLASSIFICATION_TYPE", "binary")
    if classification_type not in VALID_CLASSIFICATION_TYPES:
        raise ValueError(f"CLASSIFICATION_TYPE must be one of {VALID_CLASSIFICATION_TYPES}, got '{classification_type}'")

    model_id = os.environ.get("MODEL_ID", "traditional")
    generated_file = os.environ.get("GENERATED_FILE") or None
    generated_files = [
        value.strip()
        for value in os.environ.get("GENERATED_FILES", "").split(",")
        if value.strip()
    ]
    raw_use_llm = os.environ.get("USE_LLM", "true").strip().lower()
    if raw_use_llm not in {"true", "false"}:
        raise ValueError("USE_LLM must be either 'true' or 'false'")
    use_llm = raw_use_llm == "true"

    # REDUCTION_METHOD is kept for backward-compatibility but the pipeline ignores it —
    # the actual reduction kind is determined by the LLM-generated REDUCTION_KIND variable.
    reduction_method = os.environ.get("REDUCTION_METHOD", "selection")

    # Default to "label"; UNSW-NB15 and NSL-KDD multiclass use "attack_cat".
    target_column = os.environ.get("TARGET_COLUMN", "label")

    raw_k = os.environ.get("K_VALUES", "auto")
    if raw_k.strip().lower() == "auto":
        k_values = []
    else:
        try:
            k_values = [int(k.strip()) for k in raw_k.split(",") if k.strip()]
        except ValueError:
            raise ValueError(f"K_VALUES must be comma-separated integers, got '{raw_k}'")

    try:
        n_runs = int(os.environ.get("N_RUNS", "1"))
    except ValueError as exc:
        raise ValueError("N_RUNS must be an integer >= 1") from exc
    if n_runs < 1:
        raise ValueError("N_RUNS must be an integer >= 1")

    try:
        resampling_selection_k = int(os.environ.get("RESAMPLING_SELECTION_K", "28"))
    except ValueError as exc:
        raise ValueError("RESAMPLING_SELECTION_K must be a positive integer") from exc
    if resampling_selection_k < 1:
        raise ValueError("RESAMPLING_SELECTION_K must be a positive integer")

    if len(k_values) != len(set(k_values)) or any(k < 1 for k in k_values):
        raise ValueError("K_VALUES must contain unique positive integers")
    if model_id == "traditional" and n_runs != 1:
        raise ValueError("MODEL_ID=traditional supports only N_RUNS=1")
    if not use_llm:
        if n_runs == 1:
            if generated_files and generated_file:
                raise ValueError("Set only GENERATED_FILE or GENERATED_FILES, not both")
            if generated_files:
                if len(generated_files) != 1:
                    raise ValueError(
                        "GENERATED_FILES must contain exactly one file for N_RUNS=1"
                    )
                generated_file = generated_files[0]
        else:
            if generated_file:
                raise ValueError(
                    "Use GENERATED_FILES instead of GENERATED_FILE when N_RUNS > 1"
                )
            if len(generated_files) != n_runs or len(set(generated_files)) != n_runs:
                raise ValueError(
                    "GENERATED_FILES must contain exactly N_RUNS distinct filenames"
                )

    return ExperimentConfig(
        dataset_name=dataset_name,
        train_path=train_path,
        test_path=test_path,
        classification_type=classification_type,
        model_id=model_id,
        target_column=target_column,
        reduction_method=reduction_method,
        k_values=k_values,
        use_llm=use_llm,
        generated_file=generated_file,
        generated_files=generated_files,
        n_runs=n_runs,
        resampling_selection_k=resampling_selection_k,
    )
