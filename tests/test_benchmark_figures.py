"""Checks for the reproducible, Git-ignored comparison-figure workflow."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "generate_benchmark_figures.py"


def load_generator():
    specification = importlib.util.spec_from_file_location(
        "benchmark_figure_generator", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_retained_export_has_strictly_paired_physical_products() -> None:
    generator = load_generator()
    models = generator._load_publication_models()
    exports = generator.discover_exports(
        ROOT / "data" / "evaluations", (), models
    )
    assert [export.model_id for export in exports] == ["pino_resunet_large"]
    assert len(exports[0].domain_population) == 192
    assert len(exports[0].gauge_population) == 192 * 14
    audit = generator.validate_compatibility(exports)
    assert audit["population_and_truth_match"] is True
    assert audit["domain_trajectories"] == 192
    assert len(audit["peak_snapshots"]) == 6


def test_availability_distinguishes_missing_evaluations_from_failed_ar32() -> None:
    generator = load_generator()
    models = generator._load_publication_models()
    exports = generator.discover_exports(
        ROOT / "data" / "evaluations", (), models
    )
    table = generator.availability_table(models, exports).set_index("model_id")
    assert table.loc[
        "pino_resunet_large", "physical_evaluation_status"
    ] == "physical_evaluation_available"
    assert table.loc[
        "unet_v2", "physical_evaluation_status"
    ] == "missing_physical_evaluation"
    assert table.loc[
        "resunet_large", "physical_evaluation_status"
    ] == "unavailable_no_ar32_checkpoint"


def test_population_mismatch_fails_before_comparison() -> None:
    generator = load_generator()
    models = generator._load_publication_models()
    export = generator.discover_exports(
        ROOT / "data" / "evaluations", (), models
    )[0]
    altered = export.domain_population.iloc[:-1].copy()
    candidate = replace(
        export,
        model_id="synthetic_mismatch",
        label="Synthetic mismatch",
        domain_population=altered,
    )
    with pytest.raises(generator.ComparisonInputError, match="population differs"):
        generator.validate_compatibility((export, candidate))


def test_metric_panel_renders_from_standard_summary(tmp_path: Path) -> None:
    generator = load_generator()
    models = generator._load_publication_models()
    export = generator.discover_exports(
        ROOT / "data" / "evaluations", (), models
    )[0]
    summary = generator.combine_summary((export,), "domain")
    output = tmp_path / "domain.png"
    generator.plot_metric_panels(
        summary,
        (export,),
        generator.DOMAIN_PLOT_METRICS,
        "Test domain metrics",
        output,
        72,
    )
    assert output.is_file()
    assert output.stat().st_size > 10_000


def test_generated_figure_directory_is_always_git_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/analysis/figures/" in ignored
    assert "/analysis/.figures.render-*/" in ignored
    assert "/analysis/.figures.previous-*/" in ignored


def test_generated_figures_are_not_part_of_repository_checksums() -> None:
    checksum_script = ROOT / "scripts" / "write_repository_checksums.py"
    specification = importlib.util.spec_from_file_location(
        "repository_checksum_writer", checksum_script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    generated = ROOT / "analysis" / "figures" / "comparison_manifest.json"
    assert not module.included(generated)
    assert all(
        not path.relative_to(ROOT).is_relative_to("analysis/figures")
        for path in module.collect_paths()
    )
