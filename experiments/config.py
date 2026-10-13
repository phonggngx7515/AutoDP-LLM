"""Configuration for multiclass experiments using LLM-generated preprocessing."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

VALID_DATASETS = {"UNSW_NB15", "CIC_IDS_2017", "NSL_KDD"}


@dataclass
class ExperimentConfig:
    dataset_name: str
    train_path: str
    test_path: str
    model_id: str
    target_column: str = "attack_cat"
    classification_type: str = "multiclass"
    use_llm: bool = True
    generated_file: str | None = None
    generated_files: list[str] = field(default_factory=list)
    n_runs: int = 1

    def __post_init__(self):
        if self.dataset_name not in VALID_DATASETS:
            raise ValueError(f"DATASET_NAME must be one of {sorted(VALID_DATASETS)}")
        if self.classification_type != "multiclass":
            raise ValueError("CLASSIFICATION_TYPE must be multiclass; other tasks are unsupported.")
        if not self.model_id.strip() or self.model_id.strip().lower() == "traditional":
            raise ValueError("MODEL_ID must identify an LLM; built-in preprocessing is unsupported.")
        if not self.train_path or not self.test_path:
            raise ValueError("TRAIN_PATH and TEST_PATH must be set in .env")
        if not self.target_column:
            raise ValueError("TARGET_COLUMN must identify the multiclass target.")
        if self.dataset_name == "UNSW_NB15" and self.target_column != "attack_cat":
            raise ValueError("UNSW_NB15 multiclass requires TARGET_COLUMN=attack_cat.")
        if not isinstance(self.n_runs, int) or isinstance(self.n_runs, bool) or self.n_runs < 1:
            raise ValueError("N_RUNS must be an integer >= 1")
        if not isinstance(self.use_llm, bool):
            raise ValueError("USE_LLM must be either true or false")

        if not self.use_llm:
            if self.generated_file and self.generated_files:
                raise ValueError("Set only GENERATED_FILE or GENERATED_FILES, not both")
            if self.n_runs == 1:
                if self.generated_files:
                    if len(self.generated_files) != 1:
                        raise ValueError("GENERATED_FILES must contain exactly one filename for N_RUNS=1")
                    self.generated_file = self.generated_files[0]
                    self.generated_files = []
                if not self.generated_file:
                    raise ValueError("GENERATED_FILE must be set when USE_LLM=false and N_RUNS=1")
            else:
                if self.generated_file:
                    raise ValueError("Use GENERATED_FILES instead of GENERATED_FILE when N_RUNS > 1")
                if len(self.generated_files) != self.n_runs or len(set(self.generated_files)) != self.n_runs:
                    raise ValueError("GENERATED_FILES must contain exactly N_RUNS distinct filenames")
            filenames = self.generated_files or [self.generated_file]
            for filename in filenames:
                if Path(filename).name != filename or Path(filename).suffix != ".py":
                    raise ValueError("Generated artifacts must be Python filenames within generated/.")


def load_config() -> ExperimentConfig:
    """Load .env settings; this repository supports only the multiclass LLM workflow."""
    load_dotenv(override=True)
    dataset_name = os.environ.get("DATASET_NAME", "UNSW_NB15")
    raw_use_llm = os.environ.get("USE_LLM", "true").strip().lower()
    if raw_use_llm not in {"true", "false"}:
        raise ValueError("USE_LLM must be either true or false")
    try:
        n_runs = int(os.environ.get("N_RUNS", "1"))
    except ValueError as exc:
        raise ValueError("N_RUNS must be an integer >= 1") from exc
    default_target = "Label" if dataset_name == "CIC_IDS_2017" else "attack_cat"
    return ExperimentConfig(
        dataset_name=dataset_name,
        train_path=os.environ.get("TRAIN_PATH", ""),
        test_path=os.environ.get("TEST_PATH", ""),
        model_id=os.environ.get("MODEL_ID", ""),
        target_column=os.environ.get("TARGET_COLUMN", default_target),
        classification_type=os.environ.get("CLASSIFICATION_TYPE", "multiclass"),
        use_llm=raw_use_llm == "true",
        generated_file=os.environ.get("GENERATED_FILE") or None,
        generated_files=[value.strip() for value in os.environ.get("GENERATED_FILES", "").split(",") if value.strip()],
        n_runs=n_runs,
    )
