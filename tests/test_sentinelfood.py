from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sentinelfood as sf


def test_validate_canonical_sample() -> None:
    df = sf.load_canonical_data("examples/canonical_sample.csv")
    result = sf.validate_canonical_data(df)

    assert result["passed"] is True
    assert result["rows"] == 2
    assert result["duplicate_global_record_key"] == 0


def test_intelligence_score_is_auditable_and_sorted() -> None:
    raw = pd.DataFrame(
        {
            "name": ["low", "high", "mid"],
            "model_score_raw": [0.2, 0.9, 0.5],
            "anomaly_strength_raw": [0.1, 0.8, 0.4],
            "graph_component": [0.2, 0.9, 0.5],
        }
    )

    scored = sf.build_intelligence_score_table(raw)

    assert scored.loc[0, "name"] == "high"
    assert scored["intelligence_score_v2"].between(0, 100).all()
    points = (
        scored["score_points_model"]
        + scored["score_points_anomaly"]
        + scored["score_points_graph"]
    )
    assert np.allclose(points, scored["intelligence_score_v2"])


def test_policy_blocks_unsupported_safe_unsafe_claim() -> None:
    result = sf.deterministic_agent_answer("Bu ürün güvenli mi?")

    assert result["router_status"] == "POLICY_BLOCKED"
    assert result["policy"]["policy_code"] == "SAFE_UNSAFE_NOT_SUPPORTED"


def test_llm_validator_rejects_new_number() -> None:
    validation = sf.validate_llm_rewrite(
        llm_answer="Öncelik skoru 91 olarak hesaplandı.",
        grounded_answer="Öncelik skoru 72 olarak hesaplandı.",
    )

    assert validation["passed"] is False
    assert "numeric hallucination: 91.0" in validation["violations"]


def test_time_based_model_evaluation(tmp_path) -> None:
    rng = np.random.default_rng(42)
    train_weeks = pd.date_range("2024-01-01", periods=70, freq="W-MON")
    holdout_weeks = pd.date_range("2025-07-28", periods=25, freq="W-MON")
    weeks = train_weeks.append(holdout_weeks)
    rows = len(weeks)

    frame = pd.DataFrame(
        {
            "week": weeks,
            "future_4w_signal_target": np.tile([0, 1], rows // 2 + 1)[:rows],
            "product_canonical": np.tile(["produce", "seafood"], rows // 2 + 1)[:rows],
            "hazard_canonical": np.tile(["salmonella", "listeria"], rows // 2 + 1)[:rows],
            "count_1w": rng.integers(0, 12, size=rows),
            "mean_4w": rng.random(rows),
        }
    )

    feature_path = tmp_path / "features.csv"
    manifest_path = tmp_path / "manifest.json"
    frame.to_csv(feature_path, index=False)

    manifest = {
        "primary_model": sf.PRIMARY_MODEL,
        "primary_model_params": sf.MODEL_PARAMS,
        "primary_threshold": sf.PRIMARY_THRESHOLD,
        "secondary_watchlist_threshold": sf.SECONDARY_WATCHLIST_THRESHOLD,
        "holdout_start": sf.HOLDOUT_START,
        "holdout_end": sf.HOLDOUT_END,
        "purge_weeks_before_holdout": sf.PURGE_WEEKS,
        "categorical_features": ["product_canonical", "hazard_canonical"],
        "numeric_features": ["count_1w", "mean_4w"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = sf.evaluate_final_model(feature_path, manifest_path)

    assert result["manifest_used"] is True
    assert result["training_rows"] > 0
    assert result["holdout_rows"] > 0
    assert 0 <= result["primary_policy"]["recall"] <= 1

