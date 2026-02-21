"""CLI for the re-memory benchmark suite.

Usage:
    re-memory-bench run [benchmark] [--limit N] [--tier fast|standard|full]
    re-memory-bench download [dataset]
    re-memory-bench list
    re-memory-bench results [--latest]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.adapter import EngineAdapter
from benchmarks.adapter.metrics import BenchmarkResult

app = typer.Typer(
    name="re-memory-bench",
    help="Academic benchmark suite for re-memory engine",
    no_args_is_help=True,
)
console = Console()

RESULTS_DIR = Path(__file__).parent / "results"

BENCHMARKS = {
    "memory_agent_bench": "MemoryAgentBench (ICLR 2026) — learning, retrieval, contradiction, forgetting",
    "locomo": "LoCoMo (Snap Research) — multi-session conversations, multi-hop QA",
    "longmemeval": "LongMemEval (ICLR 2025) — temporal reasoning, knowledge updates, abstention",
    "beam": "BEAM — contradiction resolution, event ordering at scale",
    "evo_memory": "Evo-Memory (DeepMind) — streaming memory evolution",
}


def _print_results(results: list[BenchmarkResult]):
    """Print benchmark results as a Rich table."""
    for result in results:
        table = Table(
            title=f"{result.benchmark_name} — {result.task_name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Count", justify="right", style="dim")

        for metric in result.metrics:
            val_str = f"{metric.value:.4f}" if isinstance(metric.value, float) else str(metric.value)
            table.add_row(metric.name, val_str, str(metric.count) if metric.count else "")

        console.print(table)
        console.print()


def _save_results(results: list[BenchmarkResult], benchmark_name: str):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{benchmark_name}_{timestamp}.json"

    data = {
        "benchmark": benchmark_name,
        "timestamp": timestamp,
        "results": [r.to_dict() for r in results],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"  Results saved to {path}", style="green")
    return path


@app.command("list")
def list_benchmarks():
    """List available benchmarks."""
    table = Table(title="Available Benchmarks", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")

    for name, desc in BENCHMARKS.items():
        table.add_row(name, desc)

    console.print(table)


@app.command("download")
def download(
    dataset: str = typer.Argument(None, help="Dataset to download (or 'all')"),
):
    """Download benchmark datasets."""
    from benchmarks.datasets.download import download_dataset, download_all, list_datasets

    if dataset is None:
        console.print("Available datasets:", style="bold")
        list_datasets()
        return

    if dataset == "all":
        download_all()
    else:
        download_dataset(dataset)

    console.print("Done.", style="green")


@app.command("run")
def run_benchmark(
    benchmark: str = typer.Argument(..., help="Benchmark to run (or 'all')"),
    limit: int = typer.Option(None, "--limit", "-n", help="Max samples per task (for quick testing)"),
    tier: str = typer.Option("standard", "--tier", "-t", help="Observe pipeline tier: fast, standard, full"),
    tasks: str = typer.Option(None, "--tasks", help="Comma-separated task subset"),
    scale: str = typer.Option("s", "--scale", help="LongMemEval scale: s, m, l"),
    synthetic: bool = typer.Option(True, "--synthetic/--dataset", help="Use synthetic data for BEAM/Evo"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save results to file"),
    run_id: str = typer.Option(None, "--run-id", help="Unique run ID (auto-generated if not set)"),
):
    """Run a benchmark against re-memory."""
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")

    task_list = tasks.split(",") if tasks else None

    benchmarks_to_run = list(BENCHMARKS.keys()) if benchmark == "all" else [benchmark]

    for bench_name in benchmarks_to_run:
        if bench_name not in BENCHMARKS:
            console.print(f"Unknown benchmark: {bench_name}", style="red")
            console.print(f"Available: {', '.join(BENCHMARKS.keys())}")
            raise typer.Exit(1)

        console.print(f"\n{'='*70}", style="bold")
        console.print(f"  {BENCHMARKS[bench_name]}", style="bold cyan")
        console.print(f"{'='*70}", style="bold")

        # Create isolated adapter for this benchmark
        adapter = EngineAdapter.create_isolated(run_id=f"{bench_name}_{run_id}", tier=tier)

        try:
            results = _run_single(bench_name, adapter, limit=limit, tasks=task_list, scale=scale, synthetic=synthetic)

            if results:
                _print_results(results)
                if save:
                    _save_results(results, bench_name)
            else:
                console.print("  No results produced.", style="yellow")
        finally:
            adapter.close()


def _run_single(
    bench_name: str,
    adapter: EngineAdapter,
    limit: int | None = None,
    tasks: list[str] | None = None,
    scale: str = "s",
    synthetic: bool = True,
) -> list[BenchmarkResult]:
    """Dispatch to the appropriate benchmark runner."""
    if bench_name == "memory_agent_bench":
        from benchmarks.memory_agent_bench.runner import run
        return run(adapter, limit=limit, tasks=tasks)

    elif bench_name == "locomo":
        from benchmarks.locomo.runner import run
        return run(adapter, limit=limit, tasks=tasks)

    elif bench_name == "longmemeval":
        from benchmarks.longmemeval.runner import run
        return run(adapter, limit=limit, tasks=tasks, scale=scale)

    elif bench_name == "beam":
        from benchmarks.beam.runner import run
        return run(adapter, limit=limit, tasks=tasks, synthetic=synthetic)

    elif bench_name == "evo_memory":
        from benchmarks.evo_memory.runner import run
        return run(adapter, limit=limit, tasks=tasks)

    else:
        console.print(f"  No runner for {bench_name}", style="red")
        return []


@app.command("results")
def show_results(
    benchmark: str = typer.Argument(None, help="Filter by benchmark name"),
    latest: bool = typer.Option(False, "--latest", "-l", help="Show only the latest run"),
):
    """View saved benchmark results."""
    if not RESULTS_DIR.exists():
        console.print("No results directory found.", style="yellow")
        return

    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if benchmark:
        files = [f for f in files if f.name.startswith(benchmark)]

    if not files:
        console.print("No results found.", style="yellow")
        return

    if latest:
        files = files[:1]

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        console.print(f"\n{'─'*60}", style="dim")
        console.print(f"  {f.name}", style="bold")
        console.print(f"  Run: {data.get('timestamp', 'unknown')}", style="dim")

        for r in data.get("results", []):
            table = Table(
                title=f"{r['benchmark']} — {r['task']}",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            for name, info in r.get("metrics", {}).items():
                val = info.get("value", info) if isinstance(info, dict) else info
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                table.add_row(name, val_str)

            console.print(table)


@app.command("compare")
def compare_results(
    files: list[str] = typer.Argument(..., help="Result files to compare"),
):
    """Compare results across multiple benchmark runs."""
    all_data = []
    for f in files:
        path = Path(f) if Path(f).exists() else RESULTS_DIR / f
        if not path.exists():
            console.print(f"File not found: {f}", style="red")
            continue
        with open(path) as fh:
            all_data.append(json.load(fh))

    if len(all_data) < 2:
        console.print("Need at least 2 result files to compare.", style="yellow")
        return

    # Build comparison table
    # Collect all unique (task, metric) pairs
    all_keys: list[tuple[str, str]] = []
    for data in all_data:
        for r in data.get("results", []):
            for metric_name in r.get("metrics", {}):
                key = (r["task"], metric_name)
                if key not in all_keys:
                    all_keys.append(key)

    table = Table(title="Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Task / Metric", style="bold")
    for data in all_data:
        table.add_column(data.get("timestamp", "?"), justify="right")

    for task, metric in all_keys:
        values = []
        for data in all_data:
            val = None
            for r in data.get("results", []):
                if r["task"] == task and metric in r.get("metrics", {}):
                    info = r["metrics"][metric]
                    val = info.get("value", info) if isinstance(info, dict) else info
            values.append(f"{val:.4f}" if isinstance(val, float) else str(val) if val is not None else "-")

        table.add_row(f"{task} / {metric}", *values)

    console.print(table)


if __name__ == "__main__":
    app()
