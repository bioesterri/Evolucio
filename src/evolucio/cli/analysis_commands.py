"""Compact CLI views of the persisted-run analysis API."""

import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    import polars as pl

from evolucio.analysis.types import AnalysisPolicy, AnalysisWindow

analysis_app = typer.Typer(help="Analyze verified local run datasets.")

WindowSteps = Annotated[int, typer.Option("--window-steps", min=1)]
CohortWidth = Annotated[int, typer.Option("--cohort-width-steps", min=1)]
Fraction = Annotated[float, typer.Option("--bottleneck-fraction")]
StartStep = Annotated[int | None, typer.Option("--start-step", min=0)]
EndStep = Annotated[int | None, typer.Option("--end-step-exclusive", min=1)]


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


def _window(start: int | None, end: int | None) -> AnalysisWindow | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise typer.BadParameter("provide both --start-step and --end-step-exclusive")
    return AnalysisWindow(start, end)


def _view(
    section: str | None,
    run_dir: Path,
    policy: AnalysisPolicy,
    window: AnalysisWindow | None,
    output: OutputFormat,
) -> None:
    from evolucio.analysis.energy import energy_by_window
    from evolucio.analysis.population import population_by_window
    from evolucio.analysis.summary import analyze_run
    from evolucio.analysis.survival import cohort_summary
    from evolucio.analysis.windows import validated
    from evolucio.persistence.loaders.run_data import open_run

    data = validated(open_run(run_dir))
    result = analyze_run(data, policy=policy, window=window)
    payload = asdict(result)
    if section is not None:
        payload = {
            name: value
            for name, value in payload.items()
            if not name.endswith("_summary")
            or name in (f"{section}_summary", "resource_summary" if section == "energy" else "")
        }
    preview = None
    if section == "population" and result.population_summary is not None:
        preview = population_by_window(data, policy.window_steps, window).select(
            "covered_start",
            "window_end_exclusive",
            "population_start",
            "population_end",
            "births_total",
            "deaths_total",
        )
    elif section == "energy":
        preview = energy_by_window(data, policy.window_steps, window).select(
            "covered_start",
            "window_end_exclusive",
            "metric_samples",
            "stock_statistics_quality",
            "gross_energy_cost_total",
            "resource_consumed_total",
        )
    elif section == "survival" and result.survival_summary is not None:
        preview = cohort_summary(data, policy.cohort_width_steps).select(
            "cohort_start",
            "is_founder",
            "agents_total",
            "deaths_observed",
            "right_censored_count",
        )
    if preview is not None:
        sample = preview.limit(6).collect()
        payload["preview"] = sample.head(5).to_dicts()
        payload["preview_truncated"] = sample.height > 5
    if output is OutputFormat.JSON:
        typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
        return
    coverage = result.coverage
    typer.echo(f"{result.run_id} ({result.run_status})")
    typer.echo(
        f"Steps [{coverage.start_step}, {coverage.end_step_exclusive}) | "
        f"metrics interval {result.metrics_interval}"
    )
    for name, value in payload.items():
        if name.endswith("_summary") and value is not None:
            heading = name.removesuffix("_summary")
            typer.echo(
                f"{heading} [EXACT]"
                if heading in ("population", "mortality", "survival")
                else heading
            )
            for label, field in value.items():
                if isinstance(field, dict) and "quality" in field:
                    typer.echo(f"  {label} = {field['value']} [{field['quality']}]")
                elif not isinstance(field, list):
                    typer.echo(f"  {label} = {field}")
                else:
                    typer.echo(f"  {label} = {json.dumps(field, ensure_ascii=False)}")
    if result.bottleneck_episode_count is not None and section in (None, "population"):
        typer.echo(f"bottleneck_episode_count = {result.bottleneck_episode_count}")
    if preview is not None:
        for row in payload["preview"]:
            typer.echo("  " + " | ".join(f"{name} = {value}" for name, value in row.items()))
        if payload["preview_truncated"]:
            typer.echo("  … export for the full table")
    for warning in result.warnings:
        typer.echo(f"Warning | {warning}")
    for name, reason in result.unavailable_metrics.items():
        typer.echo(f"UNAVAILABLE | {name} | {reason}")


Format = Annotated[OutputFormat, typer.Option("--format")]


@analysis_app.command()
def summary(
    run_dir: Path,
    window_steps: WindowSteps = 100,
    cohort_width_steps: CohortWidth = 100,
    bottleneck_fraction: Fraction = 0.1,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize the verified run or an absolute step interval."""
    _view(
        None,
        run_dir,
        AnalysisPolicy(window_steps, cohort_width_steps, bottleneck_fraction),
        _window(start_step, end_step_exclusive),
        format,
    )


@analysis_app.command()
def population(
    run_dir: Path,
    window_steps: WindowSteps = 100,
    bottleneck_fraction: Fraction = 0.1,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize demographic changes and bottlenecks."""
    _view(
        "population",
        run_dir,
        AnalysisPolicy(window_steps, bottleneck_fraction=bottleneck_fraction),
        _window(start_step, end_step_exclusive),
        format,
    )


@analysis_app.command()
def energy(
    run_dir: Path,
    window_steps: WindowSteps = 100,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize energy and resource stocks and flows."""
    _view(
        "energy",
        run_dir,
        AnalysisPolicy(window_steps),
        _window(start_step, end_step_exclusive),
        format,
    )


@analysis_app.command()
def mortality(
    run_dir: Path,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize mortality by cause and phase."""
    _view("mortality", run_dir, AnalysisPolicy(), _window(start_step, end_step_exclusive), format)


@analysis_app.command()
def survival(
    run_dir: Path,
    cohort_width_steps: CohortWidth = 100,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize survival with right censoring."""
    _view("survival", run_dir, AnalysisPolicy(cohort_width_steps=cohort_width_steps), None, format)


@analysis_app.command("export")
def export(
    run_dir: Path,
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    window_steps: WindowSteps = 100,
    cohort_width_steps: CohortWidth = 100,
    bottleneck_fraction: Fraction = 0.1,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    include_lineages: Annotated[bool, typer.Option("--lineages")] = False,
) -> None:
    """Write available analysis tables and source provenance."""
    from evolucio.analysis.export import export_analysis
    from evolucio.persistence.loaders.run_data import open_run

    typer.echo(
        export_analysis(
            open_run(run_dir),
            output_dir,
            policy=AnalysisPolicy(window_steps, cohort_width_steps, bottleneck_fraction),
            window=_window(start_step, end_step_exclusive),
            include_lineages=include_lineages,
        )
    )


def _json_output(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))


@analysis_app.command("lineages")
def lineages(
    run_dir: Path,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
) -> None:
    """Summarize verified lineage dynamics and observed reproduction."""
    from evolucio.analysis.lineage_analysis import analyze_lineages
    from evolucio.persistence.loaders.run_data import open_run

    _json_output(
        asdict(analyze_lineages(open_run(run_dir), window=_window(start_step, end_step_exclusive)))
    )


@analysis_app.command("offspring")
def offspring(run_dir: Path) -> None:
    """Report offspring statistics including agents with no observed children."""
    from evolucio.analysis.offspring import offspring_distribution
    from evolucio.persistence.loaders.run_data import open_run

    result = offspring_distribution(open_run(run_dir))
    histogram = result.histogram.limit(101).collect()
    _json_output(
        {
            **result.summary,
            "histogram": histogram.head(100).to_dicts(),
            "histogram_truncated": histogram.height > 100,
        }
    )


@analysis_app.command("generations")
def generations(
    run_dir: Path, follow_up_steps: Annotated[int | None, typer.Option(min=1)] = None
) -> None:
    """Summarize genealogical generations with optional common reproductive follow-up."""
    from evolucio.analysis.offspring import generation_summary
    from evolucio.persistence.loaders.run_data import open_run

    _preview_table(generation_summary(open_run(run_dir), follow_up_steps=follow_up_steps))


@analysis_app.command("diversity")
def diversity(
    run_dir: Path,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    state_step: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    """Summarize genetic observations or analyze a full checkpoint."""
    from evolucio.analysis.genetic_diversity import genetic_diversity_at_checkpoint
    from evolucio.analysis.lineage_analysis import analyze_genetic_diversity, analyze_lineages
    from evolucio.analysis.windows import validated
    from evolucio.persistence.loaders.run_data import open_run

    data = validated(open_run(run_dir))
    window = _window(start_step, end_step_exclusive)
    if state_step is None:
        _json_output(
            {
                "genealogical": analyze_lineages(data, window=window).summaries.get("diversity"),
                "genetic": asdict(analyze_genetic_diversity(data, window=window)),
            }
        )
        return
    if window is not None:
        raise typer.BadParameter("--state-step cannot be combined with a metric window")
    result = genetic_diversity_at_checkpoint(data, state_step)
    payload = {
        name: getattr(result, name) for name in result.__dataclass_fields__ if name != "lineages"
    }
    _json_output(
        {
            **payload,
            "lineages": result.lineages.head(100).to_dicts(),
            "lineages_truncated": result.lineages.height > 100,
        }
    )


@analysis_app.command("mutations")
def mutations(
    run_dir: Path,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
) -> None:
    """Preview persisted mutation effects by lineage."""
    from evolucio.analysis.mutation_analysis import mutation_summary_by_lineage
    from evolucio.persistence.loaders.run_data import open_run

    _preview_table(
        mutation_summary_by_lineage(open_run(run_dir), _window(start_step, end_step_exclusive))
    )


AgentId = Annotated[int, typer.Option("--agent-id", min=0)]


@analysis_app.command("ancestors")
def ancestors(run_dir: Path, agent_id: AgentId) -> None:
    """Show the ancestor chain from the parent to the founder."""
    from evolucio.analysis.genealogy import get_ancestors
    from evolucio.persistence.loaders.run_data import open_run

    _json_output(get_ancestors(open_run(run_dir), agent_id).to_dicts())


@analysis_app.command("descendants")
def descendants(
    run_dir: Path,
    agent_id: AgentId,
    max_depth: Annotated[int | None, typer.Option(min=0)] = None,
    max_nodes: Annotated[int, typer.Option(min=1)] = 10_000,
) -> None:
    """Show descendants up to explicit depth and row limits."""
    from evolucio.analysis.genealogy import get_descendants
    from evolucio.persistence.loaders.run_data import open_run

    result = get_descendants(open_run(run_dir), agent_id, max_depth, max_nodes)
    _json_output(
        {
            "descendants": result.descendants.to_dicts(),
            "partial": result.partial,
            "direct_offspring_count": result.direct_offspring_count,
            "total_descendant_count": result.total_descendant_count,
        }
    )


@analysis_app.command("genome-distance")
def genome_distance(
    run_dir: Path,
    genome_a: Annotated[int, typer.Option("--genome-a", min=0)],
    genome_b: Annotated[int, typer.Option("--genome-b", min=0)],
    state_step: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    """Compare two identified policies from highlights or a verified checkpoint."""
    from evolucio.analysis.genome_distance import genome_distance as distance
    from evolucio.analysis.genome_distance import retained_genome
    from evolucio.analysis.types import AnalysisUnavailableError
    from evolucio.analysis.windows import validated
    from evolucio.persistence.loaders.run_data import open_run

    data = validated(open_run(run_dir))
    try:
        left = retained_genome(data, genome_a, state_step=state_step)
        right = retained_genome(data, genome_b, state_step=state_step)
    except AnalysisUnavailableError as exc:
        _json_output({"quality": "UNAVAILABLE", "reason": str(exc)})
        return
    _json_output({"quality": "EXACT", **asdict(distance(left, right))})


def _preview_table(frame: "pl.LazyFrame") -> None:
    sample = frame.limit(101).collect()
    _json_output({"rows": sample.head(100).to_dicts(), "truncated": sample.height > 100})
