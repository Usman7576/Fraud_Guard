"""`fraudguard` command line: one entry point for every pipeline stage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fraudguard import __version__
from fraudguard.config import (
    PROJECT_ROOT,
    get_settings,
    load_data_config,
    load_feature_config,
    load_model_config,
)
from fraudguard.data import pipeline, synthetic
from fraudguard.data.snapshot import SnapshotError, load_snapshot, read_manifest
from fraudguard.errors import NoChampionError
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features.registry import GROUPS, feature_columns
from fraudguard.log import setup_logging
from fraudguard.reports import adversarial

app = typer.Typer(help="FraudGuard: cost-aware fraud detection.", no_args_is_help=True)
data_app = typer.Typer(help="Data foundation: raw files to versioned snapshots.", no_args_is_help=True)
features_app = typer.Typer(help="Feature engineering.", no_args_is_help=True)
report_app = typer.Typer(help="Analysis reports.", no_args_is_help=True)
train_app = typer.Typer(help="Model training (tracked in MLflow).", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(report_app, name="report")
app.add_typer(train_app, name="train")
console = Console()

ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Path to a data config YAML.")]
FeatConfigOpt = Annotated[
    Path | None, typer.Option("--features-config", help="Path to a feature config YAML.")
]
OutOpt = Annotated[Path, typer.Option("--out", help="Folder for report files.")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    setup_logging("DEBUG" if verbose else get_settings().log_level)


@app.command()
def version() -> None:
    """Print the FraudGuard version."""
    console.print(f"fraudguard {__version__}")


@data_app.command("synth")
def data_synth(
    rows: Annotated[int, typer.Option(help="Number of synthetic transactions.")] = 20_000,
    out: Annotated[Path | None, typer.Option(help="Output folder (default: the raw data folder).")] = None,
    force: Annotated[bool, typer.Option(help="Overwrite existing raw files.")] = False,
    config: ConfigOpt = None,
) -> None:
    """Generate a small synthetic dataset shaped like IEEE-CIS (for development and CI)."""
    cfg = load_data_config(config)
    target = out or cfg.paths.raw_dir
    existing = target / cfg.raw_files.transactions
    if existing.exists() and not force:
        console.print(
            f"[red]{existing} already exists.[/] Use --force to overwrite, or --out for another folder."
        )
        raise typer.Exit(1)
    t_path, i_path = synthetic.write_synthetic(target, n_transactions=rows, seed=cfg.seed)
    console.print(f"[green]Wrote[/] {t_path} and {i_path}")


@data_app.command("build")
def data_build(
    force: Annotated[bool, typer.Option(help="Rebuild even if inputs are unchanged.")] = False,
    config: ConfigOpt = None,
) -> None:
    """Validate raw files, add time and card uid, split by time, write the base snapshot."""
    cfg = load_data_config(config)
    try:
        result = pipeline.build_base(cfg, force=force)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    table = Table(title="Time split", header_style="bold")
    for col in ("split", "rows", "fraud rate", "from", "to"):
        table.add_column(col, justify="right" if col in {"rows", "fraud rate"} else "left")
    for s in result.summaries:
        table.add_row(
            s.name, f"{s.rows:,}", f"{s.fraud_rate:.2%}", f"{s.start:%Y-%m-%d}", f"{s.end:%Y-%m-%d}"
        )
    console.print(table)
    if result.uid_consistency is not None:
        console.print(f"card_uid label consistency (train, uids with 2+ rows): {result.uid_consistency:.1%}")
    for w in result.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    status = "unchanged" if result.skipped else "written"
    console.print(f"Snapshot [bold]{pipeline.STAGE}[/] {status}, digest [cyan]{result.manifest.digest()}[/]")


@data_app.command("info")
def data_info(config: ConfigOpt = None) -> None:
    """Show the current snapshots and their manifests."""
    cfg = load_data_config(config)
    try:
        m = read_manifest(cfg.paths.snapshot_dir, pipeline.STAGE)
    except SnapshotError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"[bold]{m.stage}[/]  digest {m.digest()}  created {m.created_at}  code {m.code_version}")
    for name, entry in m.files.items():
        console.print(f"  {name}: {entry.rows:,} rows x {entry.columns} cols  sha256 {entry.sha256[:16]}...")


@app.command("all")
def run_all(
    force: Annotated[bool, typer.Option(help="Rebuild every stage.")] = False,
    config: ConfigOpt = None,
    trials: Annotated[int | None, typer.Option(help="Override the number of Optuna trials.")] = None,
) -> None:
    """Rebuild everything. Stages are added here as they are built (train, evaluate, ...)."""
    data_build(force=force, config=config)
    features_build(force=force, config=config, features_config=None)
    report_adversarial(config=config, features_config=None, out=PROJECT_ROOT / "reports")
    train_baselines(config=config, model_config=None, out=PROJECT_ROOT / "reports")
    train_lightgbm(config=config, model_config=None, out=PROJECT_ROOT / "reports", trials=trials)
    report_ablation(config=config, model_config=None, features_config=None, out=PROJECT_ROOT / "reports")
    evaluate(config=config, model_config=None, out=PROJECT_ROOT / "reports")


ModelConfigOpt = Annotated[Path | None, typer.Option("--model-config", help="Path to a model config YAML.")]


@train_app.command("baselines")
def train_baselines(
    config: ConfigOpt = None,
    model_config: ModelConfigOpt = None,
    out: OutOpt = PROJECT_ROOT / "reports",
) -> None:
    """Fit the rules baseline and logistic regression; compare them on the validation month."""
    # MLflow takes seconds to import; load it only for commands that train.
    from fraudguard.models import baselines  # noqa: PLC0415

    data_cfg, model_cfg = load_data_config(config), load_model_config(model_config)
    try:
        report = baselines.run_baselines(data_cfg, model_cfg, get_settings())
    except SnapshotError as exc:
        console.print(f"[red]{exc}[/] Run `fraudguard features build` first.")
        raise typer.Exit(1) from exc

    fpr = model_cfg.evaluation.target_fpr
    table = Table(title="Validation month", header_style="bold")
    for col in ("model", "PR-AUC", "ROC-AUC", f"recall @ {fpr:.0%} FPR", "precision there", "Brier"):
        table.add_column(col, justify="left" if col == "model" else "right")
    for r in report.results:
        v = r.validation
        brier = f"{v.brier:.4f}" if v.brier is not None else "-"
        table.add_row(
            r.name,
            f"{v.pr_auc:.4f}",
            f"{v.roc_auc:.4f}",
            f"{v.recall_at_fpr:.1%}",
            f"{v.precision_at_fpr:.1%}",
            brier,
        )
    console.print(table)
    path = baselines.write_report(report, out)
    console.print(f"Runs logged to MLflow ({get_settings().mlflow_experiment}); report at {path}")


@train_app.command("lightgbm")
def train_lightgbm(
    config: ConfigOpt = None,
    model_config: ModelConfigOpt = None,
    out: OutOpt = PROJECT_ROOT / "reports",
    trials: Annotated[int | None, typer.Option(help="Override the number of Optuna trials.")] = None,
) -> None:
    """Tune LightGBM, calibrate it, choose the cost-optimal policy and register the champion."""
    from fraudguard.models import champion  # noqa: PLC0415

    data_cfg, model_cfg = load_data_config(config), load_model_config(model_config)
    try:
        r = champion.train_champion(data_cfg, model_cfg, get_settings(), n_trials=trials)
    except SnapshotError as exc:
        console.print(f"[red]{exc}[/] Run `fraudguard features build` first.")
        raise typer.Exit(1) from exc

    v = r.validation_raw
    console.print(
        f"LightGBM validation: PR-AUC [bold]{v.pr_auc:.4f}[/], recall {v.recall_at_fpr:.1%} at "
        f"{v.target_fpr:.0%} FPR, ECE {r.ece_raw:.4f} raw -> {r.ece_calibrated:.4f} calibrated"
    )
    table = Table(title="Validation month, cost per 10,000 transactions", header_style="bold")
    for col in ("policy", "cost", "declined", "reviewed", "fraud caught"):
        table.add_column(col, justify="left" if col == "policy" else "right")
    for name, cost in (
        ("LightGBM + policy", r.model_policy["cost"]),
        ("Rules", r.rules_policy["cost"]),
        ("Approve everything", r.approve_all),
    ):
        assert isinstance(cost, dict)
        table.add_row(
            name,
            f"${cost['per_10k']:,.0f}",
            f"{cost['decline_rate']:.2%}",
            f"{cost['review_rate']:.2%}",
            f"{cost['fraud_caught']:.1%}",
        )
    console.print(table)
    status = "[green]champion[/]" if r.is_champion else "[yellow]not champion (rules are cheaper)[/]"
    console.print(
        f"Saving vs rules {r.saving_vs_rules:.1%}. "
        f"Registered {champion.REGISTERED_NAME} v{r.model_version}: {status}"
    )
    json_path, md_path = champion.write_reports(r, out)
    console.print(f"Reports at {json_path} and {md_path}")


@report_app.command("ablation")
def report_ablation(
    config: ConfigOpt = None,
    model_config: ModelConfigOpt = None,
    features_config: FeatConfigOpt = None,
    out: OutOpt = PROJECT_ROOT / "reports",
) -> None:
    """Retrain the champion on growing feature sets; PR-AUC and drift for each (validation month)."""
    from fraudguard.evaluation import ablation  # noqa: PLC0415

    data_cfg, model_cfg = load_data_config(config), load_model_config(model_config)
    try:
        r = ablation.run_ablation(data_cfg, model_cfg, load_feature_config(features_config), get_settings())
    except (SnapshotError, NoChampionError) as exc:
        console.print(f"[red]{exc}[/] Run `fraudguard train lightgbm` first.")
        raise typer.Exit(1) from exc
    table = Table(title="Feature group ablation (validation)", header_style="bold")
    for col in ("feature set", "features", "PR-AUC", "gain", "adversarial AUC"):
        table.add_column(col, justify="left" if col == "feature set" else "right")
    for row in r.rows:
        g = row.gain_pr_auc
        table.add_row(
            row.name,
            str(row.n_features),
            f"{row.pr_auc.estimate:.3f}",
            f"{g.estimate:+.3f} [{g.low:+.3f}, {g.high:+.3f}]" if g else "-",
            f"{row.adversarial_auc:.3f}" if row.adversarial_auc is not None else "-",
        )
    console.print(table)
    json_path, _ = ablation.write_reports(r, out)
    console.print(f"Report written to {json_path.parent}")


@app.command("evaluate")
def evaluate(
    config: ConfigOpt = None,
    model_config: ModelConfigOpt = None,
    out: OutOpt = PROJECT_ROOT / "reports",
) -> None:
    """Final evaluation of the champion on the untouched test month, with bootstrap intervals."""
    from fraudguard.evaluation import final  # noqa: PLC0415

    data_cfg, model_cfg = load_data_config(config), load_model_config(model_config)
    try:
        r = final.run_evaluation(data_cfg, model_cfg, get_settings())
    except (SnapshotError, NoChampionError) as exc:
        console.print(f"[red]{exc}[/] Run `fraudguard train lightgbm` first.")
        raise typer.Exit(1) from exc
    iv = r.intervals
    pct = f"{r.confidence:.0%}"
    table = Table(title=f"Test month, champion v{r.model_version} ({pct} intervals)", header_style="bold")
    for col in ("metric", "LightGBM", "rules", "difference"):
        table.add_column(col, justify="left" if col == "metric" else "right")

    def ci(name: str, fmt: str) -> str:
        v = iv[name]
        return f"{v.estimate:{fmt}} [{v.low:{fmt}}, {v.high:{fmt}}]"

    table.add_row("PR-AUC", ci("model_pr_auc", ".3f"), ci("rules_pr_auc", ".3f"), ci("diff_pr_auc", "+.3f"))
    table.add_row(
        f"recall @ {r.model_metrics.target_fpr:.0%} FPR",
        ci("model_recall_at_fpr", ".1%"),
        ci("rules_recall_at_fpr", ".1%"),
        ci("diff_recall_at_fpr", "+.1%"),
    )
    table.add_row(
        "cost per 10k",
        ci("model_cost_per_10k", ",.0f"),
        ci("rules_cost_per_10k", ",.0f"),
        ci("cost_saved_per_10k", ",.0f"),
    )
    console.print(table)
    console.print(f"Cost saving vs rules {ci('cost_saving_share', '.1%')}; ECE {r.ece:.4f}")
    json_path, md_path = final.write_reports(r, out)
    console.print(f"Reports at {json_path} and {md_path}")


@features_app.command("build")
def features_build(
    force: Annotated[bool, typer.Option(help="Rebuild even if inputs are unchanged.")] = False,
    config: ConfigOpt = None,
    features_config: FeatConfigOpt = None,
) -> None:
    """Build every feature group from the base snapshot and write the features snapshot."""
    data_cfg, feat_cfg = load_data_config(config), load_feature_config(features_config)
    try:
        result = feature_pipeline.build(data_cfg, feat_cfg, force=force)
    except SnapshotError as exc:
        console.print(f"[red]{exc}[/] Run `fraudguard data build` first.")
        raise typer.Exit(1) from exc
    for name, size in result.group_sizes.items():
        console.print(f"  {name:<10} {size:>4} columns  [dim]{GROUPS[name].description}[/]")
    status = "unchanged" if result.skipped else "written"
    console.print(
        f"Snapshot [bold]{feature_pipeline.STAGE}[/] {status}: {result.n_features} features, "
        f"digest [cyan]{result.manifest.digest()}[/]"
    )


@report_app.command("adversarial")
def report_adversarial(
    config: ConfigOpt = None,
    features_config: FeatConfigOpt = None,
    out: OutOpt = PROJECT_ROOT / "reports",
) -> None:
    """Measure how different the test period is from training (adversarial validation)."""
    data_cfg, feat_cfg = load_data_config(config), load_feature_config(features_config)
    try:
        features = load_snapshot(data_cfg.paths.snapshot_dir, feature_pipeline.STAGE, "features")
    except SnapshotError as exc:
        console.print(f"[red]{exc}[/] Run `fraudguard features build` first.")
        raise typer.Exit(1) from exc
    result = adversarial.run(features, feature_columns(features), feat_cfg.adversarial, data_cfg.seed)
    json_path, _ = adversarial.write(result, out)
    console.print(
        f"Adversarial validation AUC [bold]{result.auc_mean:.3f}[/] Â± {result.auc_std:.3f}: {result.verdict}"
    )
    for name, share in result.top_features[:5]:
        console.print(f"  {name:<28} {share:>6.1%}")
    console.print(f"Report written to {json_path.parent}")


if __name__ == "__main__":
    app()
