"""FR-021: a single CLI running the full CreditGuard chain end to end.

    python -m creditguard.pipeline.orchestrator run-all \\
        --generate --validate --clean --features --train --register --monitor

Each stage calls straight into the same functions its own phase's CLI
uses (`creditguard.data.generator.generate_dataset`,
`creditguard.validation.cli.validate_command`/`clean_command`,
`creditguard.features.build.build_features`,
`creditguard.models.train.main`, `creditguard.monitoring.scheduler.
run_monitoring_cycle`) rather than reimplementing any of them, and rather
than shelling out to each CLI as a subprocess -- direct calls keep this one
Python process, one log stream, and one place stage failures surface.

`--ingest` (loading the generated dataset into PostgreSQL -- what gives
the live application/API layer and Phase 10's monitoring something to read
from a fresh stack) is not in the brief's literal example command but is
supported as an additional stage for exactly that reason; running the
literal example command on an empty database is still safe, it just leaves
`--monitor`'s checks with nothing to report yet (handled gracefully, not as
an error -- see `monitoring/drift.py`/`data_quality.py`'s empty-window
paths).

Stages are idempotent in the sense each phase already designed for:
`--generate` is deterministic per config+seed (a rerun with the same
config produces the same `dataset_version` and safely rewrites identical
content); `--validate`/`--clean`/`--features` are pure transforms of a
`dataset_version` and simply re-derive the same output when rerun;
`--ingest` skips rows already present rather than aborting the whole run
on a duplicate-key conflict; `--train`/`--register` deliberately do NOT
no-op on rerun -- CLAUDE.md hard rule 6 requires every promotion to create
a new version, so "idempotent" here means "safe and correct to rerun," not
"a no-op"; `--monitor` only ever appends new `drift_reports`/
`monitoring_metrics`/`data_quality_issues` rows.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError

logger = logging.getLogger("creditguard.pipeline.orchestrator")

STAGE_ORDER: tuple[str, ...] = (
    "generate",
    "ingest",
    "validate",
    "clean",
    "features",
    "train",
    "register",
    "monitor",
)


class PipelineError(RuntimeError):
    """Raised when a stage fails -- always names the stage, per the Phase
    10 brief's "fails fast with a clear message naming the stage."
    """

    def __init__(self, stage: str, original: Exception) -> None:
        self.stage = stage
        self.original = original
        super().__init__(f"Pipeline stage {stage!r} failed: {original}")


@dataclass
class PipelineContext:
    """State threaded between stages -- each stage reads what an earlier
    one produced and writes what a later one needs, entirely through this
    object (never global/module state), so `--from-stage` can be given an
    explicit `dataset_version`/`clean_dataset_version` to skip straight
    into the middle of the chain.
    """

    data_generation_config_path: str = "config/data_generation.yaml"
    validation_config_path: str = "config/validation_rules.yaml"
    features_config_path: str = "config/features.yaml"
    model_config_path: str = "config/model_config.yaml"
    monitoring_config_path: str = "config/monitoring.yaml"
    n_customers: int | None = None
    seed: int | None = None
    model_families: str = "logistic,random_forest,xgboost"
    dataset_version: str | None = None
    clean_dataset_version: str | None = None
    features_output_dir: str | None = None
    registered_model_id: str | None = None
    stage_results: dict[str, Any] = field(default_factory=dict)


def _run_stage(name: str, fn: Any) -> Any:
    logger.info("stage_started", extra={"stage": name})
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: wrap and re-raise
        duration_s = time.perf_counter() - start
        logger.error(
            "stage_failed",
            extra={"stage": name, "duration_s": round(duration_s, 2)},
            exc_info=True,
        )
        raise PipelineError(name, exc) from exc
    duration_s = time.perf_counter() - start
    logger.info(
        "stage_finished", extra={"stage": name, "duration_s": round(duration_s, 2)}
    )
    print(f"[{name}] finished in {duration_s:.1f}s")
    return result


def stage_generate(ctx: PipelineContext) -> str:
    from creditguard.config import get_settings
    from creditguard.data.generator import generate_dataset, load_config
    from creditguard.data.versioning import make_dataset_version, write_dataset

    config = load_config(ctx.data_generation_config_path)
    dataset = generate_dataset(config, seed=ctx.seed, n_customers=ctx.n_customers)
    dataset_version = make_dataset_version(dataset.metadata["config"])
    write_dataset(dataset, dataset_version, get_settings().data_dir / "raw")
    ctx.dataset_version = dataset_version
    return dataset_version


def stage_ingest(ctx: PipelineContext) -> dict[str, Any]:
    from creditguard.data.ingest import ingest_dataset

    if ctx.dataset_version is None:
        raise RuntimeError("--ingest requires a dataset_version (run --generate first)")
    try:
        summary = ingest_dataset(ctx.dataset_version, truncate=False, progress=False)
    except IntegrityError:
        logger.info(
            "ingest_already_done",
            extra={"dataset_version": ctx.dataset_version},
        )
        return {"skipped": True, "reason": "already ingested"}
    return {"tables": {k: v.__dict__ for k, v in summary.tables.items()}}


def stage_validate(ctx: PipelineContext) -> dict[str, Any]:
    from creditguard.validation.cli import validate_command

    if ctx.dataset_version is None:
        raise RuntimeError(
            "--validate requires a dataset_version (run --generate first)"
        )
    args = argparse.Namespace(
        dataset_version=ctx.dataset_version,
        config=ctx.validation_config_path,
        data_dir=None,
        reports_dir=None,
    )
    result = validate_command(args)
    return {"passed": result.passed, "n_violations": len(result.violations)}


def stage_clean(ctx: PipelineContext) -> str:
    from creditguard.validation.cli import clean_command

    if ctx.dataset_version is None:
        raise RuntimeError("--clean requires a dataset_version (run --generate first)")
    output_version = f"{ctx.dataset_version}_clean"
    args = argparse.Namespace(
        dataset_version=ctx.dataset_version,
        output_version=output_version,
        config=ctx.validation_config_path,
        data_dir=None,
        output_dir=None,
        reports_dir=None,
    )
    clean_command(args)
    ctx.clean_dataset_version = output_version
    return output_version


def stage_features(ctx: PipelineContext) -> dict[str, Any]:
    from creditguard.config import get_settings
    from creditguard.features.build import build_features, load_features_config
    from creditguard.validation.engine import load_rule_config

    if ctx.clean_dataset_version is None:
        raise RuntimeError(
            "--features requires a clean dataset_version (run --clean first)"
        )
    features_config = load_features_config(ctx.features_config_path)
    cleaning_config = load_rule_config(ctx.validation_config_path)
    output_dir = Path(
        ctx.features_output_dir
        or (get_settings().data_dir / "features" / ctx.clean_dataset_version)
    )
    result = build_features(
        dataset_version=ctx.clean_dataset_version,
        split_strategy="temporal",
        output_dir=output_dir,
        features_config=features_config,
        cleaning_config=cleaning_config,
    )
    return {
        "n_features": result["metadata"]["n_features"],
        "output_dir": str(output_dir),
    }


def stage_train(ctx: PipelineContext, *, register: bool) -> dict[str, Any]:
    from creditguard.models.train import main as train_main

    if ctx.clean_dataset_version is None:
        raise RuntimeError(
            "--train requires a clean dataset_version (run --clean first)"
        )
    argv = [
        "--dataset-version",
        ctx.clean_dataset_version,
        "--models",
        ctx.model_families,
    ]
    if register:
        argv.append("--register-best")
    train_main(argv)
    return {"dataset_version": ctx.clean_dataset_version, "registered": register}


def stage_register_baseline(ctx: PipelineContext) -> dict[str, Any]:
    """Build and persist a baseline profile for the just-registered active
    model -- the natural point in the chain for "at model promotion,
    persist a baseline profile" (see `monitoring/baseline.py`'s docstring).
    """
    from creditguard.models import registry
    from creditguard.monitoring.baseline import build_and_save_baseline_for_model

    model_row = registry.get_active_model()
    if model_row is None:
        raise RuntimeError("--register requested but no active model is registered")
    ctx.registered_model_id = model_row["model_id"]
    path = build_and_save_baseline_for_model(
        model_row["model_id"],
        features_config_path=ctx.features_config_path,
        cleaning_config_path=ctx.validation_config_path,
        monitoring_config_path=ctx.monitoring_config_path,
    )
    return {"model_id": model_row["model_id"], "baseline_path": str(path)}


def stage_monitor(ctx: PipelineContext) -> dict[str, Any]:
    from creditguard.monitoring.scheduler import run_monitoring_cycle

    return run_monitoring_cycle(ctx.monitoring_config_path)


def run_all(
    *,
    stages: dict[str, bool],
    ctx: PipelineContext | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> PipelineContext:
    """Run the requested stages, in `STAGE_ORDER`, restricted to
    `[from_stage, to_stage]` if given. `stages` maps each stage name in
    `STAGE_ORDER` to whether it was requested at all (the individual
    `--generate`/`--ingest`/... flags); `from_stage`/`to_stage` further
    narrow that for a partial rerun.
    """
    ctx = ctx or PipelineContext()
    start_idx = STAGE_ORDER.index(from_stage) if from_stage else 0
    end_idx = STAGE_ORDER.index(to_stage) if to_stage else len(STAGE_ORDER) - 1

    for idx, stage_name in enumerate(STAGE_ORDER):
        if idx < start_idx or idx > end_idx:
            continue
        if not stages.get(stage_name, False):
            continue

        if stage_name == "generate":
            ctx.stage_results["generate"] = _run_stage(
                "generate", lambda: stage_generate(ctx)
            )
        elif stage_name == "ingest":
            ctx.stage_results["ingest"] = _run_stage(
                "ingest", lambda: stage_ingest(ctx)
            )
        elif stage_name == "validate":
            ctx.stage_results["validate"] = _run_stage(
                "validate", lambda: stage_validate(ctx)
            )
        elif stage_name == "clean":
            ctx.stage_results["clean"] = _run_stage("clean", lambda: stage_clean(ctx))
        elif stage_name == "features":
            ctx.stage_results["features"] = _run_stage(
                "features", lambda: stage_features(ctx)
            )
        elif stage_name == "train":
            register_requested = stages.get("register", False)
            ctx.stage_results["train"] = _run_stage(
                "train",
                lambda flag=register_requested: stage_train(ctx, register=flag),
            )
        elif stage_name == "register":
            if not stages.get("train", False):
                raise PipelineError(
                    "register",
                    RuntimeError("--register requires --train in the same run"),
                )
            ctx.stage_results["register"] = _run_stage(
                "register", lambda: stage_register_baseline(ctx)
            )
        elif stage_name == "monitor":
            ctx.stage_results["monitor"] = _run_stage(
                "monitor", lambda: stage_monitor(ctx)
            )

    return ctx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CreditGuard end-to-end pipeline orchestrator (FR-021)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-all", help="Run the full pipeline chain.")
    for stage in STAGE_ORDER:
        run_parser.add_argument(f"--{stage}", action="store_true")
    run_parser.add_argument("--from-stage", choices=STAGE_ORDER, default=None)
    run_parser.add_argument("--to-stage", choices=STAGE_ORDER, default=None)
    run_parser.add_argument("--dataset-version", default=None)
    run_parser.add_argument("--clean-dataset-version", default=None)
    run_parser.add_argument("--n-customers", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--models", default="logistic,random_forest,xgboost")
    run_parser.add_argument(
        "--data-generation-config", default="config/data_generation.yaml"
    )
    run_parser.add_argument(
        "--validation-config", default="config/validation_rules.yaml"
    )
    run_parser.add_argument("--features-config", default="config/features.yaml")
    run_parser.add_argument("--model-config", default="config/model_config.yaml")
    run_parser.add_argument("--monitoring-config", default="config/monitoring.yaml")

    return parser


def main(argv: list[str] | None = None) -> PipelineContext:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run-all":
        raise SystemExit(f"Unknown command: {args.command}")

    stages = {stage: bool(getattr(args, stage)) for stage in STAGE_ORDER}
    ctx = PipelineContext(
        data_generation_config_path=args.data_generation_config,
        validation_config_path=args.validation_config,
        features_config_path=args.features_config,
        model_config_path=args.model_config,
        monitoring_config_path=args.monitoring_config,
        n_customers=args.n_customers,
        seed=args.seed,
        model_families=args.models,
        dataset_version=args.dataset_version,
        clean_dataset_version=args.clean_dataset_version,
    )

    started_at = datetime.now(UTC)
    ctx = run_all(
        stages=stages, ctx=ctx, from_stage=args.from_stage, to_stage=args.to_stage
    )
    duration_s = (datetime.now(UTC) - started_at).total_seconds()

    print(f"\nPipeline finished in {duration_s:.1f}s")
    for stage, result in ctx.stage_results.items():
        print(f"  {stage}: {result}")
    return ctx


if __name__ == "__main__":
    main()
