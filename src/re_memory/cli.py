"""CLI entry point for re-memory.

Usage:
    re-memory setup             Set up infrastructure services
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
    re-memory purge             Wipe all memory stores
    re-memory daemon start|stop|status  Consolidation daemon
"""

from __future__ import annotations

import json
import shutil
import subprocess
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
setup_app = typer.Typer(help="Infrastructure setup")
app.add_typer(setup_app, name="setup")

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


# --- Setup subcommands ---

# Compose file embedded so `re-memory setup` works without cloning the repo
_COMPOSE_TEMPLATE = """\
services:
  {services}

volumes:
  {volumes}
"""

_SERVICE_DEFS = {
    "qdrant": (
        """\
qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5""",
        "qdrant_data:",
    ),
    "falkordb": (
        """\
falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - falkordb_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5""",
        "falkordb_data:",
    ),
    "ollama": (
        """\
ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama""",
        "ollama_data:",
    ),
    "postgres": (
        """\
postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: re_memory
      POSTGRES_USER: re_memory
      POSTGRES_PASSWORD: re_memory
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U re_memory"]
      interval: 10s
      timeout: 5s
      retries: 5""",
        "pg_data:",
    ),
}

_AVAILABLE_SERVICES = list(_SERVICE_DEFS.keys())


def _check_docker() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        console.print("[red]Docker not found. Install Docker first: https://docs.docker.com/get-docker/[/red]")
        return False
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, check=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print("[red]Docker is installed but not running. Start Docker Desktop first.[/red]")
        return False


def _generate_compose(services: list[str]) -> str:
    """Generate a docker-compose.yml for selected services."""
    svc_blocks = []
    vol_blocks = []
    for svc in services:
        defn, vol = _SERVICE_DEFS[svc]
        svc_blocks.append(defn)
        vol_blocks.append(vol)
    return _COMPOSE_TEMPLATE.format(
        services="\n  ".join("\n  ".join(b.splitlines()) for b in svc_blocks),
        volumes="\n  ".join(vol_blocks),
    )


@setup_app.callback(invoke_without_command=True)
def setup_main(
    ctx: typer.Context,
    services: Optional[list[str]] = typer.Option(
        None, "--service", "-s",
        help=f"Services to set up (choose from: {', '.join(_AVAILABLE_SERVICES)}). Repeat for multiple.",
    ),
    all_services: bool = typer.Option(False, "--all", "-a", help="Set up all services"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Set up infrastructure services via Docker.

    Select which services to start:

        re-memory setup -s qdrant -s falkordb

        re-memory setup --all

    Available services: qdrant, falkordb, ollama, postgres
    """
    global json_output
    if output_json:
        json_output = True

    if ctx.invoked_subcommand is not None:
        return

    if not _check_docker():
        raise typer.Exit(1)

    # Determine which services to start
    if all_services:
        selected = _AVAILABLE_SERVICES[:]
    elif services:
        for s in services:
            if s not in _AVAILABLE_SERVICES:
                console.print(f"[red]Unknown service: {s}. Choose from: {', '.join(_AVAILABLE_SERVICES)}[/red]")
                raise typer.Exit(1)
        selected = services
    else:
        # Interactive: show what's available and let user pick
        console.print("\n[bold]Available services:[/bold]\n")
        console.print("  [cyan]qdrant[/cyan]     Vector similarity search (recommended)")
        console.print("  [cyan]falkordb[/cyan]   Knowledge graph storage (recommended)")
        console.print("  [cyan]ollama[/cyan]     Local LLM inference (if not running natively)")
        console.print("  [cyan]postgres[/cyan]   Production event store (optional)\n")
        console.print("Usage: [bold]re-memory setup -s qdrant -s falkordb[/bold]")
        console.print("   or: [bold]re-memory setup --all[/bold]\n")
        raise typer.Exit(0)

    # Generate and write compose file
    data_dir = Path.home() / ".re-memory"
    data_dir.mkdir(parents=True, exist_ok=True)
    compose_path = data_dir / "docker-compose.yml"
    compose_content = _generate_compose(selected)
    compose_path.write_text(compose_content)

    console.print(f"\n[bold]Starting services:[/bold] {', '.join(selected)}\n")

    # Run docker compose up
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "up", "-d"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            console.print(f"[red]Docker compose failed:[/red]\n{result.stderr}")
            raise typer.Exit(1)
        console.print(result.stderr, end="")  # compose writes progress to stderr
    except subprocess.TimeoutExpired:
        console.print("[red]Docker compose timed out after 120s.[/red]")
        raise typer.Exit(1)

    # Pull embedding model if ollama was started
    if "ollama" in selected:
        console.print("\n[yellow]Pulling embedding model (mxbai-embed-large)...[/yellow]")
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "exec", "ollama",
                 "ollama", "pull", "mxbai-embed-large"],
                timeout=300,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            console.print("[yellow]Could not pull model automatically. Run manually:[/yellow]")
            console.print("  ollama pull mxbai-embed-large")

    # Summary
    summary = {"status": "running", "services": selected, "compose_file": str(compose_path)}
    if json_output:
        _output(summary)
    else:
        console.print(Panel.fit(
            "\n".join([
                f"[green]Services running:[/green] {', '.join(selected)}",
                f"[dim]Compose file: {compose_path}[/dim]",
                "",
                "Next: [bold]re-memory init[/bold]",
            ]),
            title="[bold green]Setup Complete[/bold green]",
            border_style="green",
        ))


@setup_app.command("stop")
def setup_stop():
    """Stop all infrastructure services."""
    compose_path = Path.home() / ".re-memory" / "docker-compose.yml"
    if not compose_path.exists():
        console.print("[yellow]No setup found. Run 're-memory setup' first.[/yellow]")
        raise typer.Exit(1)

    subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "down"],
        timeout=60,
    )
    console.print("[green]Services stopped.[/green]")


@setup_app.command("status")
def setup_status(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON", is_eager=True),
):
    """Check which infrastructure services are running."""
    global json_output
    if output_json:
        json_output = True

    compose_path = Path.home() / ".re-memory" / "docker-compose.yml"
    if not compose_path.exists():
        if json_output:
            _output({"status": "not_configured"})
        else:
            console.print("[yellow]No setup found. Run 're-memory setup' first.[/yellow]")
        return

    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "ps", "--format", "json"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        _output({"status": "error", "message": result.stderr.strip()})
        return

    services = []
    for line in result.stdout.strip().splitlines():
        try:
            svc = json.loads(line)
            services.append({
                "name": svc.get("Service", svc.get("Name", "?")),
                "state": svc.get("State", "?"),
                "status": svc.get("Status", "?"),
            })
        except json.JSONDecodeError:
            pass

    if json_output:
        _output({"services": services})
    else:
        _output(services, title="Infrastructure Services")


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
