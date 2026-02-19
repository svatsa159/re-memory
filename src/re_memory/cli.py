"""CLI entry point for re-memory.

Usage:
    re-memory init              Initialize memory store
    re-memory status            Memory health and statistics
    re-memory observe <input>   Store a new memory
    re-memory recall <query>    Retrieve memories
    re-memory consolidate       Trigger consolidation
    re-memory forget <id>       Forget a memory
    re-memory inspect <id>      View memory details
    re-memory search <query>    Search all layers
    re-memory history           Recent operations
    re-memory export <file>     Export memory state
    re-memory import <file>     Import memory state
    re-memory daemon start|stop|status  Consolidation daemon
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import load_config, save_config
from .engine import MemoryEngine

app = typer.Typer(
    name="re-memory",
    help="Brain-anatomical memory engine for AI agents",
)
daemon_app = typer.Typer(help="Consolidation daemon management")
app.add_typer(daemon_app, name="daemon")

console = Console()

# Global options
json_output: bool = False


def _engine() -> MemoryEngine:
    return MemoryEngine()


def _output(data: dict | list, title: str = ""):
    """Output data in JSON or Rich format."""
    if json_output:
        typer.echo(json.dumps(data, indent=2, default=str))
        return

    if isinstance(data, list):
        if not data:
            console.print("[dim]No results.[/dim]")
            return
        table = Table(title=title, show_lines=True)
        if data:
            for key in data[0].keys():
                table.add_column(key, style="cyan", overflow="fold")
            for row in data:
                table.add_row(*[str(v)[:100] for v in row.values()])
        console.print(table)
    elif isinstance(data, dict):
        table = Table(title=title, show_header=False, show_lines=True)
        table.add_column("Key", style="bold cyan", width=20)
        table.add_column("Value", style="white")
        for key, value in data.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    table.add_row(f"  {key}.{k}", str(v))
            else:
                table.add_row(key, str(value))
        console.print(table)


def _version_callback(value: bool):
    if value:
        console.print(f"re-memory v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version",
        callback=_version_callback, is_eager=True,
    ),
):
    """Brain-anatomical memory engine for AI agents."""
    global json_output
    json_output = output_json

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def init():
    """Initialize memory store for a new agent/user."""
    engine = _engine()
    results = engine.init()

    # Save default config
    config_path = save_config(engine.config)
    results["config_saved"] = str(config_path)

    if json_output:
        _output(results)
    else:
        console.print(Panel.fit(
            "\n".join(f"[cyan]{k}[/cyan]: {v}" for k, v in results.items()),
            title="[bold green]Memory System Initialized[/bold green]",
            border_style="green",
        ))


@app.command()
def status(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Memory health: counts, sizes, staleness."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    data = engine.status()
    _output(data, title="Memory System Status")


@app.command()
def observe(
    text: str = typer.Argument(..., help="Text to observe and store"),
    source: str = typer.Option("cli", "--source", "-s", help="Source of the observation"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Write path: parse, encode, and store a memory."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    result = engine.observe(text, source=source)
    _output(result, title="Memory Encoded")


@app.command()
def recall(
    query: str = typer.Argument(..., help="Query to recall memories"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-t", help="Token budget"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Read path: goal-conditioned memory retrieval."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    result = engine.recall(query, max_tokens=max_tokens, limit=limit)
    _output(result, title="Recalled Memories")


@app.command()
def consolidate(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Trigger the consolidation loop manually."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    result = engine.consolidate()
    _output(result, title="Consolidation Results")


@app.command()
def forget(
    memory_id: str = typer.Argument(..., help="Memory ID to forget"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Explicitly forget a memory by ID."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    result = engine.forget(memory_id)
    _output(result, title="Forget Result")


@app.command()
def inspect(
    memory_id: str = typer.Argument(..., help="Memory ID to inspect"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """View a specific memory with full metadata."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    result = engine.inspect(memory_id)
    if result is None:
        if json_output:
            _output({"error": "not_found", "id": memory_id})
        else:
            console.print(f"[red]Memory not found: {memory_id}[/red]")
        raise typer.Exit(1)
    _output(result, title=f"Memory: {memory_id}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Search across all memory layers."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    results = engine.search(query, limit=limit)
    _output(results, title=f"Search: {query}")


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Timeline of recent memory operations."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    results = engine.history(limit=limit)
    _output(results, title="Recent Memory Operations")


@app.command(name="export")
def export_cmd(
    path: Path = typer.Argument(..., help="Export file path"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Export memory state to a JSON file."""
    global json_output
    if output_json:
        json_output = True
    engine = _engine()
    result = engine.export_data(path)
    _output(result, title="Export Complete")


@app.command(name="import")
def import_cmd(
    path: Path = typer.Argument(..., help="Import file path"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Import memory state from a JSON file."""
    global json_output
    if output_json:
        json_output = True
    if not path.exists():
        if json_output:
            _output({"error": "file_not_found", "path": str(path)})
        else:
            console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)
    engine = _engine()
    result = engine.import_data(path)
    _output(result, title="Import Complete")


@app.command()
def purge(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Wipe all memory stores (events, vectors, graph, schemas)."""
    global json_output
    if output_json:
        json_output = True
    if not yes and not json_output:
        typer.confirm(
            "This will delete ALL memories, vectors, graph data, and schemas. Continue?",
            abort=True,
        )
    engine = _engine()
    result = engine.purge()
    _output(result, title="Purge Complete")


@app.command()
def config():
    """View current configuration."""
    cfg = load_config()
    _output(cfg.model_dump(), title="Current Configuration")


# --- Daemon subcommands ---


@daemon_app.command("start")
def daemon_start():
    """Start the background consolidation daemon."""
    console.print("[yellow]Consolidation daemon starting...[/yellow]")
    from .loops.consolidation import start_daemon

    start_daemon(_engine())


@daemon_app.command("stop")
def daemon_stop():
    """Stop the consolidation daemon."""
    from .loops.consolidation import stop_daemon

    stop_daemon()
    console.print("[green]Daemon stopped.[/green]")


@daemon_app.command("status")
def daemon_status():
    """Check consolidation daemon state."""
    from .loops.consolidation import daemon_status as _ds

    result = _ds()
    _output(result, title="Daemon Status")
