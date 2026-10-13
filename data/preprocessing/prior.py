"""
Prior data preprocessing utilities
"""

import pandas as pd
import numpy as np


def summarize_prior_preprocessing(raw_df, processed_df, label_col=None, notes=None):
    """Describe, in a fully dataset-agnostic way, what the prior preprocessing stage changed.

    Compares the RAW dataframe (as read from disk) against the PRIOR-PROCESSED one and returns a
    human-readable bullet summary for the LLM, so it does not redo already-completed cleaning.
    Best-effort: every section is guarded so this never raises and breaks generation. The optional
    ``notes`` list (per-dataset semantic hints the diff cannot infer) is appended verbatim.
    """
    lines = []
    try:
        # Normalize names (strip + lower) so whitespace/case renames don't look like drops/adds.
        raw_norm = {str(c).strip().lower(): c for c in raw_df.columns}
        proc_norm = {str(c).strip().lower(): c for c in processed_df.columns}

        if any(str(c) != str(c).strip() for c in raw_df.columns):
            lines.append("- Column names were stripped of surrounding whitespace.")

        dropped = [raw_norm[k] for k in (raw_norm.keys() - proc_norm.keys())]
        added = [proc_norm[k] for k in (proc_norm.keys() - raw_norm.keys())]
        if dropped:
            try:
                n_const = sum(1 for c in dropped if raw_df[c].nunique(dropna=False) <= 1)
            except Exception:
                n_const = 0
            extra = f" (incl. {n_const} constant/single-value)" if n_const else ""
            lines.append(f"- Dropped {len(dropped)} column(s){extra}: {sorted(map(str, dropped))[:30]}")
        if added:
            lines.append(f"- Added/derived {len(added)} column(s): {sorted(map(str, added))}")

        if len(raw_df) != len(processed_df):
            delta = len(raw_df) - len(processed_df)
            verb = "removed" if delta > 0 else "added"
            lines.append(f"- Row count: {len(raw_df)} → {len(processed_df)} "
                         f"({abs(delta)} {verb}; e.g. NaN filtering or resampling).")

        # Missing values on the columns common to both frames.
        shared = raw_norm.keys() & proc_norm.keys()
        raw_shared = [raw_norm[k] for k in shared]
        proc_shared = [proc_norm[k] for k in shared]
        raw_na = int(raw_df[raw_shared].isna().sum().sum()) if raw_shared else 0
        if raw_na > 0:
            proc_na = int(processed_df[proc_shared].isna().sum().sum()) if proc_shared else 0
            tail = (" (fully handled)." if proc_na == 0
                    else f" ({proc_na} residual NaN(s) remain — you should still handle them).")
            lines.append(f"- Missing values on shared columns: {raw_na} in raw → {proc_na} after prior{tail}")

        # Infinite values in numeric columns.
        try:
            raw_num = raw_df.select_dtypes(include=[np.number])
            raw_inf = int(np.isinf(raw_num.to_numpy()).sum()) if raw_num.shape[1] else 0
            if raw_inf > 0:
                proc_num = processed_df.select_dtypes(include=[np.number])
                proc_inf = int(np.isinf(proc_num.to_numpy()).sum()) if proc_num.shape[1] else 0
                lines.append(f"- Infinite values in numeric columns: {raw_inf} in raw → {proc_inf} after prior.")
        except Exception:
            pass

        if label_col and label_col in processed_df.columns:
            uniq = pd.unique(processed_df[label_col].dropna())
            kind = f"{len(uniq)} classes"
            lines.append(f"- Target column '{label_col}' is present "
                         f"(dtype={processed_df[label_col].dtype}, {kind}).")
    except Exception as e:  # pragma: no cover - defensive; never break generation
        lines.append(f"- (Automatic prior-diff partially failed: {e})")

    if notes:
        lines.append("- Dataset-specific notes:")
        lines.extend(f"    • {n}" for n in notes)

    if not lines:
        return "No structural changes detected between the raw and prior-processed data."
    return "\n".join(lines)


class UNSW_NB15_PriorPreprocessor:

    # Optional semantic notes for the LLM (things the auto-diff cannot infer). Keyed by
    # classification type; consumed via summarize_prior_preprocessing(..., notes=...).
    PRIOR_NOTES = {
        "multiclass": [
            "Dropped 'id' and 'label' (record identifier and the binary target).",
            "In categorical 'service', the placeholder '-' was replaced with 'other'.",
        ],
    }

    @staticmethod
    def multi_class_prior_preprocessing(dataset_df: pd.DataFrame) -> pd.DataFrame:
        """Initial preprocessing: drop certain columns and handle missing"""
        dataset = dataset_df.drop(['id', 'label'], axis=1)
        dataset['service'] = dataset['service'].replace('-', 'other')
        return dataset


class CIC_IDS_2017_PriorPreprocessor:

    PRIOR_NOTES = {
        "multiclass": [
            'Stripped whitespace from column names.',
            'Replaced ±infinity with NaN and dropped rows containing NaN.',
            "Dropped 10 all-zero/bulk-flag columns (e.g. 'Bwd PSH Flags', 'Fwd URG Flags', 'CWE Flag Count', and the Avg-Bytes/Packets/Bulk-Rate columns).",
        ],
    }

    @staticmethod
    def multi_class_prior_preprocessing(dataset_df: pd.DataFrame) -> pd.DataFrame:
        """Initial preprocessing: drop certain columns and handle missing"""
        dataset = dataset_df.replace([np.inf, -np.inf], np.nan).copy()
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        dataset.columns = dataset.columns.str.strip()
        dataset = dataset.drop(columns=[
            "Bwd PSH Flags",
            "Fwd URG Flags",
            "Bwd URG Flags",
            "CWE Flag Count",
            "Fwd Avg Bytes/Bulk",
            "Bwd Avg Bytes/Bulk",
            "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate"
        ])
        return dataset

class NSL_KDD_PriorPreprocessor:

    PRIOR_NOTES = {
        "multiclass": [
            'Stripped whitespace from column names.',
            "Dropped 'difficulty' and the constant 'num_outbound_cmds' feature.",
            "Mapped raw attack names into the five-class 'attack_cat' target.",
        ],
    }

    @staticmethod
    def multi_class_prior_preprocessing(
        dataset_df: pd.DataFrame,
        drop_num_outbound_cmds: bool = True,
    ) -> pd.DataFrame:
        """Initial preprocessing: handle missing and format labels for NSL-KDD into 5 standard classes"""
        dataset = dataset_df.copy()
        dataset.columns = dataset.columns.str.strip()

        if 'difficulty' in dataset.columns:
            dataset = dataset.drop('difficulty', axis=1)

        if drop_num_outbound_cmds and 'num_outbound_cmds' in dataset.columns:
            dataset = dataset.drop('num_outbound_cmds', axis=1)

        # Standard NSL-KDD 5-class mapping
        attack_map = {
            # DoS
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
            'teardrop': 'DoS', 'apache2': 'DoS', 'mailbomb': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
            # Probe
            'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
            # R2L
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'R2L', 'imap': 'R2L',
            'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L', 'sendmail': 'R2L',
            'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
            'warezmaster': 'R2L', 'worm': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
            # U2R
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'ps': 'U2R',
            'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
            # Normal
            'normal': 'normal'
        }

        label_col = 'label' if 'label' in dataset.columns else 'class'
        if label_col in dataset.columns:
            # Map raw labels to the 5 categories
            dataset['attack_cat'] = dataset[label_col].map(attack_map).fillna(dataset[label_col])
            if label_col != 'attack_cat':
                dataset = dataset.drop(label_col, axis=1)

        return dataset
