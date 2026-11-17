from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fraudguard.config import (
    AblationConfig,
    DataConfig,
    FinalEvaluationConfig,
    LightGBMConfig,
    Settings,
    load_feature_config,
    load_model_config,
)
from fraudguard.data import pipeline as data_pipeline
from fraudguard.errors import NoChampionError
from fraudguard.evaluation import ablation, final
from fraudguard.evaluation.bootstrap import bootstrap, resamples
from fraudguard.explain import reasons
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features.registry import group_of
from fraudguard.models import champion


class TestBootstrap:
    def test_interval_covers_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(10, 2, 2_000)
        iv = bootstrap(
            lambda i: {"mean": float(x[i].mean())},
            x.size,
            groups=None,
            n_resamples=500,
            confidence=0.95,
            seed=1,
        )["mean"]
        assert iv.low < 10 < iv.high
        assert iv.high - iv.low == pytest.approx(2 * 1.96 * 2 / math.sqrt(2_000), rel=0.2)
        assert iv.n_valid == 500

    def test_day_clusters_are_resampled_whole(self) -> None:
        groups = np.repeat(np.arange(10), 5)
        for idx in resamples(groups, groups.size, 20, seed=0):
            assert idx.size == groups.size
            counts = np.bincount(groups[idx], minlength=10)
            assert set(counts.tolist()) <= {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}

    def test_paired_difference_is_tighter_than_either_side(self) -> None:
        rng = np.random.default_rng(2)
        shared = rng.normal(0, 5, 3_000)  # noise common to both models
        a, b = shared + 0.2 + rng.normal(0, 0.5, 3_000), shared + rng.normal(0, 0.5, 3_000)

        def stats(i: np.ndarray) -> dict[str, float]:
            return {"a": float(a[i].mean()), "b": float(b[i].mean()), "diff": float((a[i] - b[i]).mean())}

        ivs = bootstrap(stats, 3_000, groups=None, n_resamples=300, confidence=0.95, seed=3)
        assert ivs["diff"].excludes_zero()
        assert ivs["diff"].high - ivs["diff"].low < (ivs["a"].high - ivs["a"].low) / 3
        assert not (ivs["a"].low > ivs["b"].high)  # separate intervals overlap: unpaired would miss it

    def test_undefined_draws_are_skipped(self) -> None:
        y = np.array([0] * 99 + [1])
        iv = bootstrap(
            lambda i: {"rate": float(y[i].mean()) if y[i].any() else float("nan")},
            y.size,
            groups=None,
            n_resamples=200,
            confidence=0.9,
            seed=0,
        )["rate"]
        assert 0 < iv.n_valid < 200


class TestReasons:
    def test_descriptions(self) -> None:
        assert reasons.describe("vel_card_1h_cnt", 7) == "7 earlier transactions on this card in the last 1h"
        assert reasons.describe("vel_device_24h_amt", 1234.5) == "$1,234 spent on this device in the last 24h"
        assert reasons.describe("graph_card_known_fraud", 1) == "card has 1 confirmed fraud chargeback"
        assert reasons.describe("graph_card_n_addresses", 2) == "card used with 2 billing addresses"
        assert (
            reasons.describe("card_secs_since_last", 7_200)
            == "2 hours since this card's previous transaction"
        )
        assert reasons.describe("DeviceInfo", None) == "device information is missing"
        assert reasons.describe("hour", 23.5) == "made at 23:30"
        assert reasons.describe("weekday", 7) == "made on a Sunday"
        assert "masked" in reasons.describe("V258", 1.0)
        assert reasons.describe("id_31", None).endswith("missing (meaning masked by the data provider)")
        assert reasons.describe("something_new", 3) == "something_new is 3"
        assert (
            reasons.describe("vel_device_7d_cnt", 0)
            == "no earlier transactions on this device in the last 7d"
        )
        assert reasons.describe("amt_cents", 0.0) == "round amount (no cents)"
        assert reasons.describe("amt_cents", 0.99) == "amount ends in .99"
        assert reasons.describe("amt_cents", 0.418).startswith("amount ends in .418, a fraction of a cent")
        assert reasons.describe("D1", 0.0) == "card first seen today"
        assert reasons.describe("D1", 14.0) == "card first seen 14 days ago"

    def test_only_risk_raising_features_are_reasons(self) -> None:
        df = pl.DataFrame(
            {"card_prior_txn": [0, 5], "TransactionAmt": [900.0, 10.0], "card_is_new": [True, False]}
        )
        contrib = np.array([[0.2, 1.5, 0.1], [-0.3, -0.1, 0.4]])
        out = reasons.top_reasons(contrib, df, df.columns, k=2)
        assert [r.feature for r in out[0]] == ["TransactionAmt", "card_prior_txn"]
        assert out[0][0].text == "amount of $900.00"
        assert [r.feature for r in out[1]] == ["card_is_new"]
        assert out[1][0].text == "card seen before"

    def test_features_that_read_the_same_are_one_reason(self) -> None:
        df = pl.DataFrame({"TransactionAmt": [849.0], "amt_log": [np.log1p(849.0)], "D1": [0.0], "C1": [3.0]})
        contrib = np.array([[0.9, 0.4, 0.6, 0.5]])
        out = reasons.top_reasons(contrib, df, df.columns, k=3)[0]
        assert [r.text for r in out] == ["amount of $849.00", "card first seen today", out[2].text]
        assert out[0].feature == "TransactionAmt" and out[0].contribution == 1.3
        assert out[2].feature == "C1"

    def test_global_importance_ranks_by_mean_absolute_value(self) -> None:
        contrib = np.array([[0.1, -2.0], [0.1, 2.0]])
        assert [n for n, _ in reasons.global_importance(contrib, ["a", "b"])] == ["b", "a"]


def test_feature_groups_and_ablation_variants() -> None:
    cols = [
        "TransactionAmt",
        "V1",
        "vel_card_1h_cnt",
        "card_prior_txn",
        "card_amt_z",
        "graph_2hop_cards",
        "D1",
    ]
    assert [group_of(c) for c in cols] == [
        "base",
        "base",
        "velocity",
        "behaviour",
        "behaviour",
        "graph",
        "base",
    ]
    names = [(n, len(c)) for n, c in ablation.variants(cols, ["card_prior_txn", "D1", "graph_2hop_cards"])]
    assert names == [
        ("base", 3),
        ("+ velocity", 4),
        ("+ behaviour", 6),
        ("+ graph", 7),
        ("all - lifetime counters", 4),
    ]
    assert len(ablation.variants(cols, [])) == 4


def test_amount_bands_are_labelled_in_order() -> None:
    assert final.amount_band(np.array([10.0, 50.0, 99.0, 5000.0]), [50, 100]) == [
        "< $50",
        "$50-100",
        "$50-100",
        ">= $100",
    ]


def test_missing_champion_is_a_clear_error(tmp_path: Path) -> None:
    uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    with pytest.raises(NoChampionError):
        champion.champion_version(uri)


@pytest.mark.slow
def test_evaluation_and_ablation_end_to_end(data_config: DataConfig) -> None:
    data_pipeline.build_base(data_config)
    feature_pipeline.build(data_config, load_feature_config())
    tmp = data_config.paths.raw_dir.parent
    settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", mlflow_experiment="t"
    )
    cfg = load_model_config().model_copy(
        update={
            "lightgbm": LightGBMConfig(n_trials=2, n_folds=2, early_stopping_rounds=20, max_rounds=200),
            "evaluation_final": FinalEvaluationConfig(n_resamples=40),
            "ablation": AblationConfig(
                drift_prone=["card_prior_txn", "D1"], n_resamples=20, adversarial=False
            ),
        }
    )
    trained = champion.train_champion(data_config, cfg, settings)
    if not trained.is_champion:
        pytest.skip("synthetic run did not beat the rules baseline, so there is no champion to evaluate")

    model = champion.load_champion(settings.mlflow_tracking_uri)
    assert model.params == trained.tuning["best_params"] and model.rounds == trained.tuning["best_rounds"]

    # SHAP values add up to the model's log-odds.
    features = pl.read_parquet(data_config.paths.snapshot_dir / "features" / "features.parquet").head(300)
    contrib, base = model.contributions(features)
    raw = model.raw_score(features)
    assert np.allclose(1 / (1 + np.exp(-(contrib.sum(axis=1) + base))), raw, atol=1e-6)

    report = final.run_evaluation(data_config, cfg, settings)
    iv = report.intervals
    assert iv["model_pr_auc"].low <= iv["model_pr_auc"].estimate <= iv["model_pr_auc"].high
    assert iv["diff_pr_auc"].estimate == pytest.approx(
        iv["model_pr_auc"].estimate - iv["rules_pr_auc"].estimate, abs=1e-5
    )
    assert report.costs["model"].n == report.n_rows
    assert sum(s.n_fraud for s in report.error_analysis if s.segment == "amount") == report.n_fraud
    assert report.examples and all(len(e["reasons"]) <= 3 for e in report.examples)  # type: ignore[arg-type]
    j, m = final.write_reports(report, tmp / "reports")
    assert json.loads(j.read_text())["model_version"] == report.model_version
    assert "Error analysis" in m.read_text(encoding="utf-8")

    abl = ablation.run_ablation(data_config, cfg, load_feature_config(), settings)
    assert [r.name for r in abl.rows][-1] == "all - lifetime counters"
    assert abl.rows[0].gain_pr_auc is None and all(r.gain_pr_auc is not None for r in abl.rows[1:])
    assert abl.rows[3].n_features == len(model.feature_names)
    ablation.write_reports(abl, tmp / "reports")
