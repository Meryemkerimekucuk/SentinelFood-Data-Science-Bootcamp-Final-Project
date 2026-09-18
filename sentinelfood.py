from __future__ import annotations

"""
SENTINELFOOD AI — CONSOLIDATED PROJECT SUBMISSION
=================================================

Bu dosya, SentinelFood AI projesinin form yüklemesi için hazırlanmış
TEK-DOSYA teslim sürümüdür.

Orijinal proje modülerdir. Başlıca modüller:
- FDA / RASFF veri alma ve doğrulama
- canonical dönüşüm ve merge
- EDA
- leakage-safe zaman özellikleri
- Logistic Regression final modeli
- untouched holdout değerlendirmesi
- graph context
- IntelligenceScore
- deterministic Agent + policy guard
- guarded local LLM rewrite
- FastAPI servis katmanı

Bu tek dosya, değerlendirme sırasında ana metodolojinin okunabilmesi için
kritik parçaları bir araya getirir. Deneysel ablation, audit ve eski sürüm
scriptleri özellikle dahil edilmemiştir.

ÖNEMLİ SEMANTİK SINIRLAR
------------------------
1) model_score gerçek dünya hastalık / toksisite olasılığı değildir.
2) IntelligenceScore risk yüzdesi değildir.
3) Sistem "güvenli / güvensiz" gıda kararı vermez.
4) Graph ilişkisi nedensellik kanıtı değildir.
5) Final model/threshold kararları holdout açılmadan önce dondurulmuştur.

Final frozen policy:
- Model              : Logistic Regression
- class_weight       : balanced
- solver             : liblinear
- max_iter           : 3000
- random_state       : 42
- primary threshold  : 0.59
- watchlist threshold: 0.45
- holdout            : 2025-07-28 -> 2026-07-20
- purge              : 4 weeks

IntelligenceScore:
- 60% model component
- 20% anomaly component
- 20% graph component
"""

import argparse
import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# OPTIONAL ML IMPORTS
# ---------------------------------------------------------------------

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =====================================================================
# 1. PROJECT CONSTANTS
# =====================================================================

PROJECT_NAME = "SentinelFood AI"
SUBMISSION_VERSION = "SENTINELFOOD_CONSOLIDATED_SUBMISSION_V1"

TARGET_COL = "future_4w_signal_target"
WEEK_COL = "week"

PRIMARY_MODEL = "LOGISTIC_REGRESSION"
PRIMARY_THRESHOLD = 0.59
SECONDARY_WATCHLIST_THRESHOLD = 0.45

HOLDOUT_START = "2025-07-28"
HOLDOUT_END = "2026-07-20"
PURGE_WEEKS = 4

MODEL_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 3000,
    "random_state": 42,
    "solver": "liblinear",
}

INTELLIGENCE_WEIGHTS = {
    "model": 0.60,
    "anomaly": 0.20,
    "graph": 0.20,
}

CANONICAL_COLUMNS = [
    "alert_id",
    "event_id",
    "event_date",
    "recall_initiation_date",
    "source",
    "country",
    "origin_country",
    "state",
    "product_raw",
    "product_category",
    "hazard_raw",
    "hazard",
    "hazard_category",
    "severity_or_class",
    "reason_text",
    "distribution_pattern",
    "status",
    "recalling_firm",
]

GLOBAL_KEY = "global_record_key"

SAFETY_FOOTER = (
    "Not: SentinelFood operasyonel önceliklendirme için geliştirilmiştir. "
    "IntelligenceScore risk yüzdesi değildir; model_score gerçek dünya "
    "olasılığı değildir ve sistem safe/unsafe kararı vermez."
)


# =====================================================================
# 2. CANONICAL DATA VALIDATION
# =====================================================================

def load_canonical_data(csv_path: str | Path) -> pd.DataFrame:
    """
    SentinelFood canonical veri setini yükler ve temel şema kontrollerini yapar.

    Beklenen yapı:
        global_record_key + 18 canonical alan
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Canonical veri seti bulunamadı: {csv_path}")

    df = pd.read_csv(
        csv_path,
        dtype={
            GLOBAL_KEY: "string",
            "alert_id": "string",
            "event_id": "string",
        },
        low_memory=False,
    )

    required = [GLOBAL_KEY, *CANONICAL_COLUMNS]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Canonical veri setinde eksik kolonlar var: "
            + ", ".join(missing)
        )

    return df


def validate_canonical_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Şema, global key, kaynak ve tarih alanları için temel kalite kontrolü.
    """
    result: dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "schema_ok": True,
    }

    duplicate_global_key = int(df[GLOBAL_KEY].duplicated().sum())
    result["duplicate_global_record_key"] = duplicate_global_key

    expected_key = (
        df["source"].astype("string")
        + "::"
        + df["alert_id"].astype("string")
    )

    result["global_key_pattern_mismatch"] = int(
        (expected_key != df[GLOBAL_KEY].astype("string")).sum()
    )

    parsed_event_date = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    result["event_date_parse_failures"] = int(parsed_event_date.isna().sum())

    result["source_counts"] = {
        str(k): int(v)
        for k, v in df["source"].value_counts(dropna=False).items()
    }

    result["date_min"] = (
        None
        if parsed_event_date.dropna().empty
        else str(parsed_event_date.min().date())
    )

    result["date_max"] = (
        None
        if parsed_event_date.dropna().empty
        else str(parsed_event_date.max().date())
    )

    result["passed"] = bool(
        duplicate_global_key == 0
        and result["global_key_pattern_mismatch"] == 0
        and result["event_date_parse_failures"] == 0
    )

    return result


def canonical_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Kaynak bazlı hızlı EDA özeti.
    """
    out: dict[str, Any] = {}

    out["source_counts"] = (
        df["source"]
        .value_counts(dropna=False)
        .rename_axis("source")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    for source_name in ["FDA", "RASFF"]:
        source_df = df.loc[df["source"] == source_name].copy()

        out[f"{source_name.lower()}_rows"] = len(source_df)

        if len(source_df):
            out[f"{source_name.lower()}_missingness_pct"] = {
                col: round(float(source_df[col].isna().mean() * 100), 2)
                for col in [
                    "product_raw",
                    "product_category",
                    "hazard_raw",
                    "hazard",
                    "hazard_category",
                    "origin_country",
                    "reason_text",
                    "severity_or_class",
                ]
                if col in source_df.columns
            }

    return out


# =====================================================================
# 3. FINAL LOGISTIC REGRESSION PIPELINE
# =====================================================================

def _require_sklearn() -> None:
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "Bu mod için scikit-learn gerekli. "
            "Kurulum: pip install scikit-learn"
        )


def make_onehot():
    _require_sklearn()

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        # Eski sklearn sürümleriyle uyumluluk
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any] | None = None,
):
    """
    Final model ailesiyle uyumlu preprocessing + Logistic Regression pipeline.
    """
    _require_sklearn()

    params = dict(MODEL_PARAMS if model_params is None else model_params)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                make_onehot(),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(**params)

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def threshold_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Threshold bazlı metrikleri üretir.

    NOT:
    y_score calibrated gerçek dünya probability olarak yorumlanmaz.
    """
    _require_sklearn()

    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    prevalence = float(pd.Series(y_true).mean())

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "rows": len(y_true),
        "positive": int((pd.Series(y_true) == 1).sum()),
        "negative": int((pd.Series(y_true) == 0).sum()),
        "prevalence": prevalence,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f2": float(
            fbeta_score(
                y_true,
                y_pred,
                beta=2,
                zero_division=0,
            )
        ),
        "predicted_positive_rate": float(y_pred.mean()),
        "alerts_generated": int(y_pred.sum()),
        "precision_lift_over_prevalence": (
            float(precision / prevalence)
            if prevalence > 0
            else math.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _safe_roc_auc(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> float:
    _require_sklearn()

    if pd.Series(y_true).nunique() < 2:
        return math.nan

    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> float:
    _require_sklearn()

    if int((pd.Series(y_true) == 1).sum()) == 0:
        return math.nan

    return float(average_precision_score(y_true, y_score))


def _infer_submission_features(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Yalnız tek-dosya teslim sürümü için fallback feature seçimi.

    Gerçek final proje çalıştırmasında frozen manifestteki feature whitelist
    kullanılmalıdır. Burada manifest yoksa target/week/future alanları dışlanır,
    product_canonical ve hazard_canonical kategorik kabul edilir.
    """
    categorical = [
        col
        for col in ["product_canonical", "hazard_canonical"]
        if col in df.columns
    ]

    reserved = {
        TARGET_COL,
        WEEK_COL,
        "current_operational_signal",
    }

    numeric = []

    for col in df.columns:
        if col in reserved or col in categorical:
            continue

        if col.startswith("future_") or "target" in col.casefold():
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)

    return numeric, categorical


def evaluate_final_model(
    feature_csv: str | Path,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """
    Frozen final yaklaşımı zaman ayrımlı train/purge/holdout yapısıyla uygular.

    Eğer frozen manifest verilirse:
        feature listesi + model parametreleri manifestten okunur.

    Manifest verilmezse:
        bu tek-dosya submission sürümü güvenli fallback feature seçimi yapar.
        Resmi proje performans iddiası için manifestli çalışma tercih edilmelidir.
    """
    _require_sklearn()

    feature_csv = Path(feature_csv)

    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV bulunamadı: {feature_csv}")

    df = pd.read_csv(feature_csv, low_memory=False)

    required = [WEEK_COL, TARGET_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Model feature dosyasında eksik zorunlu kolonlar: "
            + ", ".join(missing)
        )

    df[WEEK_COL] = pd.to_datetime(
        df[WEEK_COL],
        errors="coerce",
    )

    if df[WEEK_COL].isna().any():
        raise ValueError("week alanında parse edilemeyen tarih bulundu.")

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="raise",
    ).astype(int)

    model_params = dict(MODEL_PARAMS)
    primary_threshold = PRIMARY_THRESHOLD
    watchlist_threshold = SECONDARY_WATCHLIST_THRESHOLD
    holdout_start = pd.Timestamp(HOLDOUT_START)
    holdout_end = pd.Timestamp(HOLDOUT_END)
    purge_weeks = PURGE_WEEKS

    manifest_used = False

    if manifest_json is not None:
        manifest_path = Path(manifest_json)

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Frozen manifest bulunamadı: {manifest_path}"
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            frozen = json.load(f)

        if frozen.get("primary_model") != PRIMARY_MODEL:
            raise RuntimeError(
                "Frozen manifestte beklenen model LOGISTIC_REGRESSION değil."
            )

        model_params = dict(
            frozen.get(
                "primary_model_params",
                MODEL_PARAMS,
            )
        )

        primary_threshold = float(
            frozen.get(
                "primary_threshold",
                PRIMARY_THRESHOLD,
            )
        )

        watchlist_threshold = float(
            frozen.get(
                "secondary_watchlist_threshold",
                SECONDARY_WATCHLIST_THRESHOLD,
            )
        )

        holdout_start = pd.Timestamp(
            frozen.get(
                "holdout_start",
                HOLDOUT_START,
            )
        )

        holdout_end = pd.Timestamp(
            frozen.get(
                "holdout_end",
                HOLDOUT_END,
            )
        )

        purge_weeks = int(
            frozen.get(
                "purge_weeks_before_holdout",
                PURGE_WEEKS,
            )
        )

        categorical_features = list(
            frozen["categorical_features"]
        )
        numeric_features = list(
            frozen["numeric_features"]
        )

        manifest_used = True

    else:
        (
            numeric_features,
            categorical_features,
        ) = _infer_submission_features(df)

    feature_cols = [
        *categorical_features,
        *numeric_features,
    ]

    forbidden = [
        col
        for col in feature_cols
        if (
            col.startswith("future_")
            or "target" in col.casefold()
        )
    ]

    if forbidden:
        raise RuntimeError(
            "Leakage guard failed. Yasak feature: "
            + ", ".join(forbidden)
        )

    missing_features = [
        col
        for col in feature_cols
        if col not in df.columns
    ]

    if missing_features:
        raise KeyError(
            "Feature dosyasında bulunmayan kolonlar: "
            + ", ".join(missing_features)
        )

    if not feature_cols:
        raise ValueError("Kullanılabilir model feature bulunamadı.")

    train_cutoff_exclusive = (
        holdout_start
        - pd.Timedelta(weeks=purge_weeks)
    )

    train_df = df.loc[
        df[WEEK_COL] < train_cutoff_exclusive
    ].copy()

    holdout_df = df.loc[
        (df[WEEK_COL] >= holdout_start)
        & (df[WEEK_COL] <= holdout_end)
    ].copy()

    if train_df.empty:
        raise ValueError("Training bölümü boş kaldı.")

    if holdout_df.empty:
        raise ValueError("Holdout bölümü boş kaldı.")

    pipeline = make_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )

    pipeline.fit(
        train_df[feature_cols],
        train_df[TARGET_COL],
    )

    # predict_proba burada decision/ranking score üretmek için kullanılır.
    # class-weight nedeniyle calibrated real-world probability iddiası yapılmaz.
    y_score = pipeline.predict_proba(
        holdout_df[feature_cols]
    )[:, 1]

    y_true = holdout_df[TARGET_COL]

    primary = threshold_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=primary_threshold,
    )

    watchlist = threshold_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=watchlist_threshold,
    )

    return {
        "project": PROJECT_NAME,
        "submission_version": SUBMISSION_VERSION,
        "manifest_used": manifest_used,
        "model": PRIMARY_MODEL,
        "model_params": model_params,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "training_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(holdout_end.date()),
        "purge_weeks": int(purge_weeks),
        "primary_policy": primary,
        "secondary_watchlist_policy": watchlist,
        "ranking_metrics": {
            "average_precision_pr_auc": _safe_average_precision(
                y_true,
                y_score,
            ),
            "roc_auc": _safe_roc_auc(
                y_true,
                y_score,
            ),
        },
        "interpretation": (
            "model_score ranking/decision score olarak kullanılır; "
            "calibrated gerçek dünya olasılığı değildir."
        ),
    }


# =====================================================================
# 4. INTELLIGENCE SCORE
# =====================================================================

def percentile_rank_high_good(series: pd.Series) -> pd.Series:
    """
    Yüksek değerin daha yüksek öncelik anlamına geldiği percentile rank.
    Çıktı yaklaşık 0..1 aralığındadır.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().all():
        return pd.Series(
            np.zeros(len(numeric)),
            index=numeric.index,
            dtype=float,
        )

    numeric = numeric.fillna(numeric.median())

    return numeric.rank(
        method="average",
        pct=True,
        ascending=True,
    )


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:
    total = float(sum(weights.values()))

    if total <= 0:
        raise ValueError("Score ağırlıkları toplamı pozitif olmalı.")

    return {
        key: float(value / total)
        for key, value in weights.items()
    }


def compute_intelligence_score(
    model_component: pd.Series,
    anomaly_component: pd.Series,
    graph_component: pd.Series,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    SentinelFood IntelligenceScore birleşimi.

    Beklenen component'ler 0..1 ölçeğindedir.
    Sonuç 0..100 operational prioritization score'dur.

    Bu skor:
    - toksikolojik risk değildir,
    - hastalık olasılığı değildir,
    - safe/unsafe kararı değildir.
    """
    w = normalize_weights(
        INTELLIGENCE_WEIGHTS
        if weights is None
        else weights
    )

    score = 100.0 * (
        w["model"] * model_component.astype(float)
        + w["anomaly"] * anomaly_component.astype(float)
        + w["graph"] * graph_component.astype(float)
    )

    return score


def build_intelligence_score_table(
    df: pd.DataFrame,
    model_score_col: str = "model_score_raw",
    anomaly_col: str = "anomaly_strength_raw",
    graph_component_col: str = "graph_component",
) -> pd.DataFrame:
    """
    Operational snapshot üzerinde birleşik skor üretir.

    model score -> percentile rank
    anomaly     -> mevcut 0..1 component
    graph       -> mevcut 0..1 component
    """
    required = [
        model_score_col,
        anomaly_col,
        graph_component_col,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "IntelligenceScore için eksik kolonlar: "
            + ", ".join(missing)
        )

    out = df.copy()

    out["model_component"] = percentile_rank_high_good(
        out[model_score_col]
    )

    out["anomaly_component"] = (
        pd.to_numeric(
            out[anomaly_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )

    out["graph_component"] = (
        pd.to_numeric(
            out[graph_component_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )

    out["intelligence_score_v2"] = compute_intelligence_score(
        model_component=out["model_component"],
        anomaly_component=out["anomaly_component"],
        graph_component=out["graph_component"],
    )

    w = normalize_weights(INTELLIGENCE_WEIGHTS)

    out["score_points_model"] = (
        100 * w["model"] * out["model_component"]
    )

    out["score_points_anomaly"] = (
        100 * w["anomaly"] * out["anomaly_component"]
    )

    out["score_points_graph"] = (
        100 * w["graph"] * out["graph_component"]
    )

    out["priority_rank_all"] = (
        out["intelligence_score_v2"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return out.sort_values(
        "intelligence_score_v2",
        ascending=False,
    ).reset_index(drop=True)


# =====================================================================
# 5. AGENT POLICY / SAFETY LAYER
# =====================================================================

def _normalize_text(text: str) -> str:
    value = str(text).strip().casefold().replace("ı", "i")
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
    )

    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9%]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


def policy_guard_user_query(
    query: str,
) -> dict[str, Any]:
    """
    Kullanıcı sorgusunu Agent'a göndermeden önce güvenlik açısından kontrol eder.
    """
    q = _normalize_text(query)

    safe_unsafe_patterns = [
        "guvenli mi",
        "guvensiz mi",
        "safe mi",
        "unsafe mi",
        "yenir mi",
        "tuketilir mi",
        "tuketmek guvenli",
        "kesin tehlikeli",
        "kesin guvenli",
    ]

    if any(p in q for p in safe_unsafe_patterns):
        return {
            "allowed": False,
            "policy_code": "SAFE_UNSAFE_NOT_SUPPORTED",
            "message": (
                "SentinelFood bir safe/unsafe karar sistemi değildir. "
                "Operational önceliklendirme ve sinyal bağlamı sunabilir; "
                "bir gıdanın kesin güvenli veya güvensiz olduğunu söyleyemez."
            ),
        }

    if (
        ("intelligencescore" in q or "intelligence score" in q)
        and (
            "risk" in q
            or "%" in q
            or "yuzde" in q
            or "olaslik" in q
        )
    ):
        return {
            "allowed": False,
            "policy_code": "INTELLIGENCE_SCORE_NOT_RISK_PERCENT",
            "message": (
                "IntelligenceScore bir risk yüzdesi veya hastalık "
                "olasılığı değildir. Sadece operational öncelik "
                "sıralamasında kullanılan birleşik skordur."
            ),
        }

    if (
        (
            "model score" in q
            or "modelscore" in q
            or "model skoru" in q
        )
        and (
            "%" in q
            or "yuzde" in q
            or "olaslik" in q
            or "probability" in q
        )
    ):
        return {
            "allowed": False,
            "policy_code": "MODEL_SCORE_NOT_PROBABILITY",
            "message": (
                "model_score calibrated gerçek dünya olasılığı değildir. "
                "Class-weighted Logistic Regression tarafından üretilen "
                "ranking/decision skorudur."
            ),
        }

    country_patterns = [
        "hangi ulke",
        "hangi ulkeden",
        "mensei",
        "origin country",
        "country",
        "ulke bilgisi",
    ]

    if any(p in q for p in country_patterns):
        return {
            "allowed": False,
            "policy_code": "COUNTRY_NOT_AVAILABLE",
            "message": (
                "Mevcut operational panelde origin_country bilgisi "
                "güvenilir biçimde kullanılabilir değilse Agent ülke "
                "ilişkisi çıkarmaz veya uydurmaz."
            ),
        }

    brand_patterns = [
        "hangi marka",
        "hangi sirket",
        "hangi firma",
        "markasi ne",
        "marka baglantisi",
        "sirket baglantisi",
    ]

    if any(p in q for p in brand_patterns):
        return {
            "allowed": False,
            "policy_code": "BRAND_COMPANY_GENERALIZATION_NOT_SUPPORTED",
            "message": (
                "Product-Hazard graph coarse ürün gruplarıyla çalışır. "
                "Marka veya şirket ilişkisi graph'tan genellenemez."
            ),
        }

    causality_patterns = [
        "neden oldu",
        "sebep oldu",
        "sebebi bu mu",
        "buna yol acti",
        "caused by",
    ]

    if any(p in q for p in causality_patterns):
        return {
            "allowed": False,
            "policy_code": "CAUSALITY_NOT_SUPPORTED",
            "message": (
                "SentinelFood graph ve model çıktıları gözlenmiş ilişki "
                "ve operasyonel sinyal bağlamı sunar; biyolojik veya "
                "epidemiyolojik nedensellik kanıtlamaz."
            ),
        }

    return {
        "allowed": True,
        "policy_code": "ALLOW",
        "message": "Query policy check passed.",
    }


# =====================================================================
# 6. GUARDED LLM REWRITE VALIDATOR
# =====================================================================

LLM_SYSTEM_RULES = """
Sen SentinelFood AI için yalnızca dilsel yeniden yazım yapan bir açıklama katmanısın.

Kesin kurallar:
- GROUNDED_ANSWER dışına çıkma.
- Yeni veri, yeni sayı, yeni ilişki, yeni kaynak veya URL üretme.
- GROUNDED_ANSWER içindeki sayıları değiştirme.
- score_points değerlerini yüzdeye çevirme.
- IntelligenceScore'u risk yüzdesi veya hastalık olasılığı gibi sunma.
- model_score'u gerçek dünya probability gibi sunma.
- safe/unsafe kararı verme.
- Graph ilişkisini nedensellik gibi sunma.
- Marka, şirket veya ülke ilişkisi uydurma.
""".strip()


def _extract_numbers(text: str) -> list[float]:
    values: list[float] = []

    for raw in re.findall(
        r"(?<![\w])[-+]?\d+(?:[.,]\d+)?",
        str(text),
    ):
        try:
            values.append(
                float(raw.replace(",", "."))
            )
        except ValueError:
            pass

    return values


def validate_llm_rewrite(
    llm_answer: str,
    grounded_answer: str,
) -> dict[str, Any]:
    """
    Numeric/unit drift için post-validator.

    Temel kural:
    LLM grounded cevapta olmayan yeni sayı veya yüzde üretmemeli.
    """
    violations: list[str] = []

    if "%" in llm_answer and "%" not in grounded_answer:
        violations.append(
            "unit drift: LLM introduced percent sign"
        )

    grounded_numbers = _extract_numbers(grounded_answer)
    answer_numbers = _extract_numbers(llm_answer)

    for value in answer_numbers:
        if not any(
            abs(value - g) < 1e-9
            for g in grounded_numbers
        ):
            violations.append(
                f"numeric hallucination: {value}"
            )

    answer_norm = _normalize_text(llm_answer)

    unsafe_assertions = [
        "kesin guvenli",
        "kesin guvensiz",
        "risk yuzdesidir",
        "hastalik olasiligidir",
    ]

    for phrase in unsafe_assertions:
        if phrase in answer_norm:
            violations.append(
                f"unsafe semantic assertion: {phrase}"
            )

    violations = list(dict.fromkeys(violations))

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "grounded_numbers": grounded_numbers,
        "answer_numbers": answer_numbers,
    }


def call_local_lmstudio_rewrite(
    *,
    model: str,
    user_query: str,
    grounded_answer: str,
    lmstudio_root: str = "http://127.0.0.1:1234",
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """
    Local LM Studio'ya yalnız dilsel rewrite görevi gönderir.
    """
    endpoint = (
        lmstudio_root.rstrip("/")
        + "/api/v1/chat"
    )

    body = {
        "model": model,
        "input": (
            "USER_QUERY:\n"
            f"{user_query}\n\n"
            "GROUNDED_ANSWER:\n"
            f"{grounded_answer}\n\n"
            "Görev: Yalnızca GROUNDED_ANSWER içeriğini daha doğal "
            "Türkçeyle yeniden ifade et. Yeni bilgi veya sayı ekleme."
        ),
        "system_prompt": LLM_SYSTEM_RULES,
        "stream": False,
        "temperature": 0.0,
        "max_output_tokens": 240,
        "reasoning": "off",
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    token = os.getenv("SENTINELFOOD_LLM_API_KEY")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        url=endpoint,
        data=json.dumps(
            body,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=int(timeout_seconds),
        ) as response:
            raw = response.read().decode("utf-8")

    except (
        TimeoutError,
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as exc:
        raise RuntimeError(
            f"Local LLM bağlantı hatası: {exc}"
        ) from exc

    parsed = json.loads(raw)

    texts: list[str] = []

    for item in parsed.get("output", []):
        if item.get("type") == "message":
            content = item.get("content")

            if isinstance(content, str):
                texts.append(content)

    answer = "\n".join(
        text.strip()
        for text in texts
        if text and text.strip()
    ).strip()

    if not answer:
        raise RuntimeError(
            "LM Studio cevap verdi ancak message text bulunamadı."
        )

    validation = validate_llm_rewrite(
        llm_answer=answer,
        grounded_answer=grounded_answer,
    )

    return {
        "accepted": bool(validation["passed"]),
        "answer": (
            answer
            if validation["passed"]
            else grounded_answer
        ),
        "raw_llm_answer": answer,
        "post_validation": validation,
    }


# =====================================================================
# 7. SIMPLE DETERMINISTIC AGENT ENTRY
# =====================================================================

def deterministic_agent_answer(
    query: str,
) -> dict[str, Any]:
    """
    Tek-dosya submission sürümündeki en temel Agent giriş noktası.

    Orijinal projede router 8 tool arasında seçim yapar:
    - risk_score_tool
    - graph_search_tool
    - explanation_tool
    - trend_analysis_tool
    - prediction_tool
    - source_lookup_tool
    - sql_query_tool
    - report_generator

    Burada tool dosyaları tek tek gömülmediği için policy davranışı ve
    güvenli fallback gösterilir.
    """
    policy = policy_guard_user_query(query)

    if not policy["allowed"]:
        return {
            "agent_version": SUBMISSION_VERSION,
            "router_status": "POLICY_BLOCKED",
            "policy": policy,
            "answer": policy["message"],
        }

    return {
        "agent_version": SUBMISSION_VERSION,
        "router_status": "NEEDS_PROJECT_TOOL_DATA",
        "policy": policy,
        "answer": (
            "Sorgu policy kontrolünden geçti. Tam SentinelFood projesinde "
            "deterministic router uygun analitik tool'u seçer. "
            + SAFETY_FOOTER
        ),
    }


# =====================================================================
# 8. OPTIONAL FASTAPI SERVICE
# =====================================================================

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:

    app = FastAPI(
        title="SentinelFood AI — Submission API",
        version="1.0.0",
        description=(
            "Operational food-safety intelligence submission API. "
            "Scores are not toxicological risk percentages."
        ),
    )

    class AgentQueryRequest(BaseModel):
        query: str = Field(
            ...,
            min_length=1,
            max_length=2000,
        )

    @app.get("/health")
    def api_health() -> dict[str, Any]:
        return {
            "service": SUBMISSION_VERSION,
            "status": "ok",
            "semantics": {
                "intelligence_score": (
                    "Operational prioritization only; not a risk percentage."
                ),
                "model_score": (
                    "Ranking/decision score; not calibrated probability."
                ),
                "safe_unsafe": (
                    "SentinelFood does not issue safe/unsafe determinations."
                ),
            },
        }

    @app.post("/agent/query")
    def api_agent_query(
        request: AgentQueryRequest,
    ) -> dict[str, Any]:
        return deterministic_agent_answer(
            query=request.query
        )


# =====================================================================
# 9. CLI
# =====================================================================

def _print_json(value: Any) -> None:
    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SentinelFood AI consolidated project submission."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "info",
            "validate",
            "model",
            "score",
            "policy",
        ],
        default="info",
    )

    parser.add_argument(
        "--canonical",
        help="sentinelfood_canonical.csv yolu",
    )

    parser.add_argument(
        "--features",
        help="person3_features_train_presignal_v1.csv yolu",
    )

    parser.add_argument(
        "--manifest",
        help="Frozen model policy JSON yolu",
    )

    parser.add_argument(
        "--score-input",
        help=(
            "model_score_raw, anomaly_strength_raw ve graph_component "
            "kolonlarını içeren operational CSV"
        ),
    )

    parser.add_argument(
        "--score-output",
        default="sentinelfood_intelligence_score_output.csv",
    )

    parser.add_argument(
        "--query",
        help="Policy/Agent testi için kullanıcı sorgusu",
    )

    args = parser.parse_args()

    if args.mode == "info":
        _print_json(
            {
                "project": PROJECT_NAME,
                "version": SUBMISSION_VERSION,
                "final_model": {
                    "family": PRIMARY_MODEL,
                    "params": MODEL_PARAMS,
                    "primary_threshold": PRIMARY_THRESHOLD,
                    "watchlist_threshold": SECONDARY_WATCHLIST_THRESHOLD,
                    "holdout": [
                        HOLDOUT_START,
                        HOLDOUT_END,
                    ],
                    "purge_weeks": PURGE_WEEKS,
                },
                "intelligence_score_weights": INTELLIGENCE_WEIGHTS,
                "safety_footer": SAFETY_FOOTER,
            }
        )
        return

    if args.mode == "validate":
        if not args.canonical:
            parser.error(
                "--mode validate için --canonical gerekli."
            )

        df = load_canonical_data(args.canonical)

        _print_json(
            {
                "validation": validate_canonical_data(df),
                "eda": canonical_eda_summary(df),
            }
        )
        return

    if args.mode == "model":
        if not args.features:
            parser.error(
                "--mode model için --features gerekli."
            )

        result = evaluate_final_model(
            feature_csv=args.features,
            manifest_json=args.manifest,
        )

        _print_json(result)
        return

    if args.mode == "score":
        if not args.score_input:
            parser.error(
                "--mode score için --score-input gerekli."
            )

        score_df = pd.read_csv(
            args.score_input,
            low_memory=False,
        )

        scored = build_intelligence_score_table(
            score_df
        )

        output = Path(args.score_output)

        scored.to_csv(
            output,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Kaydedildi: {output.resolve()}")
        return

    if args.mode == "policy":
        if not args.query:
            parser.error(
                "--mode policy için --query gerekli."
            )

        _print_json(
            deterministic_agent_answer(
                query=args.query
            )
        )
        return


if __name__ == "__main__":
    main()
