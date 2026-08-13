"""Tests for creditguard.pipeline.orchestrator -- the FR-021 CLI itself.

Every stage function here is monkeypatched to a cheap stand-in, since each
one's real behaviour (real generation, real training, ...) is already
covered by its own phase's test suite (and, for the full real chain,
`test_end_to_end.py`). What this file actually exercises is `orchestrator`'s
own control flow: stage ordering, `--from-stage`/`--to-stage` restriction,
`--register` requiring `--train` in the same run, and that a stage failure
is wrapped in a `PipelineError` naming the stage -- including through
`main()` itself, the one entry point no other Phase 10 test calls directly
(see `docs/FRD_acceptance_checklist.md`'s FR-021 note).
"""

from __future__ import annotations

import pytest

from creditguard.pipeline import orchestrator


def _stub_all_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def _make(name: str, result: object = None):
        def _stage(ctx, **kwargs):  # noqa: ARG001 - ctx required by call sites
            calls.append(name)
            return result if result is not None else {"stage": name}

        return _stage

    monkeypatch.setattr(orchestrator, "stage_generate", _make("generate", "ds_fake"))
    monkeypatch.setattr(orchestrator, "stage_ingest", _make("ingest"))
    monkeypatch.setattr(orchestrator, "stage_validate", _make("validate"))
    monkeypatch.setattr(orchestrator, "stage_clean", _make("clean", "ds_fake_clean"))
    monkeypatch.setattr(orchestrator, "stage_features", _make("features"))

    def _stage_train(ctx, *, register):
        calls.append("train")
        return {"registered": register}

    monkeypatch.setattr(orchestrator, "stage_train", _stage_train)
    monkeypatch.setattr(orchestrator, "stage_register_baseline", _make("register"))
    monkeypatch.setattr(orchestrator, "stage_monitor", _make("monitor"))


def test_run_all_only_runs_requested_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _stub_all_stages(monkeypatch, calls)

    stages = {name: False for name in orchestrator.STAGE_ORDER}
    stages["generate"] = True
    stages["validate"] = True

    orchestrator.run_all(stages=stages)

    assert calls == ["generate", "validate"]


def test_run_all_respects_from_and_to_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _stub_all_stages(monkeypatch, calls)

    stages = dict.fromkeys(orchestrator.STAGE_ORDER, True)
    ctx = orchestrator.PipelineContext(
        dataset_version="ds_existing", clean_dataset_version="ds_existing_clean"
    )

    orchestrator.run_all(
        stages=stages, ctx=ctx, from_stage="clean", to_stage="features"
    )

    assert calls == ["clean", "features"]


def test_run_all_register_without_train_raises_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _stub_all_stages(monkeypatch, calls)

    stages = {name: False for name in orchestrator.STAGE_ORDER}
    stages["register"] = True  # deliberately without "train"

    with pytest.raises(orchestrator.PipelineError) as exc_info:
        orchestrator.run_all(stages=stages)

    assert exc_info.value.stage == "register"


def test_run_all_train_and_register_together_passes_register_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _stub_all_stages(monkeypatch, calls)

    stages = {name: False for name in orchestrator.STAGE_ORDER}
    stages["train"] = True
    stages["register"] = True

    ctx = orchestrator.PipelineContext(clean_dataset_version="ds_existing_clean")
    result_ctx = orchestrator.run_all(stages=stages, ctx=ctx)

    assert calls == ["train", "register"]
    assert result_ctx.stage_results["train"]["registered"] is True


def test_run_stage_wraps_failure_in_pipeline_error_naming_the_stage() -> None:
    def _boom():
        raise ValueError("synthetic failure")

    with pytest.raises(orchestrator.PipelineError) as exc_info:
        orchestrator._run_stage("validate", _boom)

    assert exc_info.value.stage == "validate"
    assert isinstance(exc_info.value.original, ValueError)


def test_main_invokes_run_all_through_the_real_cli_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one path no other Phase 10 test exercises: `orchestrator.main()`
    itself, parsing real `argv` the way a user's shell invocation would.
    """
    calls: list[str] = []
    _stub_all_stages(monkeypatch, calls)

    ctx = orchestrator.main(
        [
            "run-all",
            "--generate",
            "--validate",
            "--clean",
            "--features",
            "--train",
            "--register",
            "--monitor",
            "--models",
            "logistic_regression",
        ]
    )

    assert calls == [
        "generate",
        "validate",
        "clean",
        "features",
        "train",
        "register",
        "monitor",
    ]
    assert set(ctx.stage_results) == set(calls)
    printed = capsys.readouterr().out
    assert "Pipeline finished" in printed
