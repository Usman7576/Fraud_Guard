"""`fraudguard` command line: one entry point for every pipeline stage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fraudguard import __version__
from fraudguard.config import PROJECT_ROOT, get_settings, load_data_config, load_feature_config
from fraudguard.data import pipeline, synthetic
from fraudguard.data.snapshot import SnapshotError, load_snapshot, read_manifest
from fraudguard.features import pipeline as feature_pipeline
from fraudguard.features.registry import GROUPS, feature_columns
from fraudguard.log import setup_logging
from fraudguard.reports import adversarial

app = typer.Typer(help="FraudGuard: cost-aware fraud detection.", no_args_is_help=True)
data_app = typer.Typer(help="Data foundation: raw files to versioned snapshots.", no_args_is_help=True)
features_app = typer.Typer(help="Feature engineering.", no_args_is_help=True)
report_app = typer.Typer(help="Analysis reports.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(report_app, name="report")
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
) -> None:
    """Rebuild everything. Stages are added here as they are built (train, evaluate, ...)."""
    data_build(force=force, config=config)
    features_build(force=force, config=config, features_config=None)
    report_adversarial(config=config, features_config=None, out=PROJECT_ROOT / "reports")


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
        f"Adversarial validation AUC [bold]{result.auc_mean:.3f}[/] ± {result.auc_std:.3f}: {result.verdict}"
    )
    for name, share in result.top_features[:5]:
        console.print(f"  {name:<28} {share:>6.1%}")
    console.print(f"Report written to {json_path.parent}")


if __name__ == "__main__":
    app()
