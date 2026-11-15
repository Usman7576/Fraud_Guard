from __future__ import annotations

import json

import mlflow
import numpy as np
import polars as pl
import pytest

from fraudguard.config import DataConfig, RuleConfig, Settings, load_feature_config, load_model_config
from fraudguard.data import pipeline as data_pipeline
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.models import baselines
from fraudguard.models.metrics import evaluate, recall_precision_at_fpr, threshold_at_fpr
from fraudguard.models.rules import RulesBaseline


class TestMetrics:
    def test_threshold_respects_fpr_budget(self) -> None:
        y = np.array([0] * 100 + [1] * 10)
        score = np.concatenate([np.linspace(0, 1, 100), np.full(10, 2.0)])
        t = threshold_at_fpr(y, score, 0.05)
        assert ((score[y == 0] > t).mean()) <= 0.05
        recall, precision, _ = recall_precision_at_fpr(y, score, 0.05)
        assert recall == 1.0 and precision > 0.6

    def test_ties_are_handled_conservatively(self) -> None:
        # Every genuine row has the same score as half the fraud: none can be flagged within 1% FPR.
        y = np.array([0] * 100 + [1] * 10)
        score = np.array([0.5] * 100 + [0.5] * 5 + [0.9] * 5)
        recall, _, _ = recall_precision_at_fpr(y, score, 0.01)
        assert recall == 0.5

    def test_random_scores_are_near_chance(self) -> None:
        rng = np.random.default_rng(0)
        y = (rng.random(20_000) < 0.035).astype(int)
        m = evaluate(y, rng.random(20_000), 0.01)
        assert 0.45 < m.roc_auc < 0.55
        assert m.pr_auc < 0.06
        assert m.recall_at_fpr < 0.04

    def test_non_probability_scores_skip_calibration_metrics(self) -> None:
        y = np.array([0, 0, 1, 1])
        m = evaluate(y, np.array([0.0, 1.0, 2.0, 3.0]), 0.5, is_probability=False)
        assert m.brier is None and "brier" not in m.as_dict()

    def test_single_class_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="both"):
            evaluate(np.zeros(5), np.zeros(5))


class TestRules:
    def _df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "TransactionAmt": [50.0, 2000.0, 400.0],
                "card_is_new": [False, False, True],
                "vel_card_1h_cnt": [0, 5, None],
                "isFraud": [0, 1, 1],
            }
        )

    def test_score_is_weighted_sum_of_fired_rules(self) -> None:
        rules = RulesBaseline(
            [
                RuleConfig(name="big", feature="TransactionAmt", op=">", value=1000, weight=1.0),
                RuleConfig(name="burst", feature="vel_card_1h_cnt", op=">=", value=3, weight=2.0),
                RuleConfig(name="new_high", feature="card_amt_new_high", op=">=", value=1, weight=0.5),
            ]
        )
        assert rules.score(self._df()).tolist() == [0.0, 3.0, 0.5]  # null velocity does not fire
        stats = {s.name: s for s in rules.rule_stats(self._df())}
        assert stats["big"].fired_share == pytest.approx(1 / 3, abs=1e-4)
        assert stats["big"].fraud_rate_when_fired == 1.0

    def test_unknown_feature_is_an_error(self) -> None:
        rules = RulesBaseline([RuleConfig(name="x", feature="nope", op=">", value=1)])
        with pytest.raises(KeyError, match="nope"):
            rules.score(self._df())


@pytest.mark.slow
def test_baselines_end_to_end(data_config: DataConfig) -> None:
    data_pipeline.build_base(data_config)
    feature_pipeline.build(data_config, load_feature_config())
    tmp = data_config.paths.raw_dir.parent
    settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", mlflow_experiment="test"
    )

    report = baselines.run_baselines(data_config, load_model_config(), settings)
    by_name = {r.name: r for r in report.results}
    assert set(by_name) == {"rules", "logistic_regression"}
    lr = by_name["logistic_regression"].validation
    assert lr.pr_auc > lr.fraud_rate * 3  # far better than chance on the synthetic signal
    assert lr.brier is not None
    assert by_name["rules"].details["rule_stats"]

    path = baselines.write_report(report, tmp / "reports")
    saved = json.loads(path.read_text())
    assert saved["features_digest"] == report.features_digest

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert {r.info.run_name for r in runs} == {"rules_baseline", "logistic_regression"}
    assert all(r.data.tags["features_digest"] == report.features_digest for r in runs)
