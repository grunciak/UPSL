from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


TARGET_CANDIDATES = ["quality", "Quality", "target", "Target", "label", "Label"]

NUMERIC_DTYPES = ("int64", "int32", "float64", "float32")


@dataclass
class DatasetInfo:
    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_cols: List[str]
    target_col: str
    missing_total: int
    duplicates: int


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Common in WineQT: an "Id" column that is just a row identifier.
    for c in ["Id", "id", "ID"]:
        if c in df.columns:
            df = df.drop(columns=[c])
            break
    return df


def infer_target(df: pd.DataFrame) -> str:
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    # fallback: last column
    return df.columns[-1]


def get_dataset_info(df: pd.DataFrame, target_col: str) -> DatasetInfo:
    numeric_cols = [c for c in df.columns if df[c].dtype.name in NUMERIC_DTYPES]
    missing_total = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    return DatasetInfo(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
        numeric_cols=numeric_cols,
        target_col=target_col,
        missing_total=missing_total,
        duplicates=duplicates,
    )


def make_filters(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    st.sidebar.subheader("Filtry danych")
    filtered = df.copy()

    # Categorical filters (small cardinality)
    cat_cols = []
    for c in df.columns:
        if c in numeric_cols:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 25:
            cat_cols.append(c)

    if cat_cols:
        with st.sidebar.expander("Filtry kategoryczne", expanded=False):
            for c in cat_cols:
                options = sorted([x for x in df[c].dropna().unique().tolist()])
                sel = st.multiselect(f"{c}", options=options, default=options)
                filtered = filtered[filtered[c].isin(sel)]

    with st.sidebar.expander("Filtry liczbowe", expanded=True):
        for c in numeric_cols:
            if c not in filtered.columns:
                continue
            col = filtered[c].dropna()
            if col.empty:
                continue
            lo, hi = float(col.min()), float(col.max())
            if np.isclose(lo, hi):
                continue
            step = (hi - lo) / 100.0
            rng = st.slider(
                f"{c}",
                min_value=float(lo),
                max_value=float(hi),
                value=(float(lo), float(hi)),
                step=float(step) if step > 0 else 0.01,
            )
            filtered = filtered[(filtered[c] >= rng[0]) & (filtered[c] <= rng[1])]

    return filtered


def target_mode(df: pd.DataFrame, target_col: str) -> Tuple[pd.Series, str]:
    """
    Return y and a label describing target mode.
    - "multiclass": keep original quality levels
    - "binary_good": quality >= threshold
    """
    st.sidebar.subheader("Ustawienia celu (target)")
    mode = st.sidebar.radio(
        "Tryb predykcji",
        options=["Wieloklasowy (oryginalne oceny)", "Binarny: wino dobre vs reszta"],
        index=0,
    )
    y = df[target_col]
    if mode.startswith("Binarny"):
        thr = st.sidebar.slider("Próg jakości (>=)", min_value=int(y.min()), max_value=int(y.max()), value=7)
        y = (y >= thr).astype(int)
        return y, f"binary_good_>={thr}"
    return y, "multiclass"


def build_model(kind: str, params: Dict[str, Any]) -> Pipeline:
    if kind == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=int(params.get("max_iter", 2000)),
            C=float(params.get("C", 1.0)),
            solver="lbfgs",
            n_jobs=None,
            class_weight=params.get("class_weight", None),
            multi_class="auto",
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if kind == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth", None) in (None, 0, "None") else int(params["max_depth"]),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
            class_weight=params.get("class_weight", None),
        )
        return Pipeline([("clf", clf)])

    raise ValueError(f"Unknown model kind: {kind}")


@st.cache_resource(show_spinner=False)
def train_model_cached(X: pd.DataFrame, y: pd.Series, kind: str, params_json: str, test_size: float, seed: int):
    params = __import__("json").loads(params_json)
    model = build_model(kind, params)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )
    model.fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    avg = "binary" if y_test.nunique() == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average=avg)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
    }


def cross_val_predictions(X: pd.DataFrame, y: pd.Series, kind: str, params: Dict[str, Any], folds: int, seed: int):
    model = build_model(kind, params)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = cross_val_predict(model, X, y, cv=cv)
    return preds


def feature_importance(model: Pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
    if "clf" not in model.named_steps:
        return None
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(clf, "coef_"):
        coefs = clf.coef_
        if coefs.ndim == 2:
            vals = np.mean(np.abs(coefs), axis=0)
        else:
            vals = np.abs(coefs)
        imp = pd.DataFrame({"feature": feature_names, "importance": vals})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)

    return None
