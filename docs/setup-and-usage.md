# Setup & Usage Guide

## Install

### Via pipx (recommended)

```bash
pipx install re-memory
```

This installs the `re-memory` command globally in an isolated environment. No venv management needed.

### Via pip

```bash
pip install re-memory
```

### From source (development)

```bash
git clone <repo-url>
cd re-memory
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Optional: Rust accelerator

The Rust extension provides faster math for pattern separation, Hopfield recall, and decay computation. It's **completely optional** — all functions have pure-Python fallbacks.

```bash
# Requires Rust toolchain: https://rustup.rs
pip install maturin
maturin develop --release
```

---

## Infrastructure Setup

re-memory uses external services for vector search, knowledge graphs, and LLM inference. All are optional — see [Graceful Degradation](#what-if-services-arent-running).

### Using `re-memory setup` (recommended)

Pick only the services you need:

```bash
# Recommended minimum: vector search + knowledge graph
re-memory setup -s qdrant -s falkordb

# Everything including local LLM and production database
re-memory setup --all

# Check what's running
re-memory setup status

# Stop services
re-memory setup stop
```

Available services:

| Service | Flag | Port | Purpose |
|:---|:---|:---|:---|
| **qdrant** | `-s qdrant` | 6333 | Vector similarity search for memory retrieval |
| **falkordb** | `-s falkordb` | 6379 | Knowledge graph for structured facts |
| **ollama** | `-s ollama` | 11434 | Local LLM and embedding inference |
| **postgres** | `-s postgres` | 5432 | Production-grade event store (replaces SQLite) |

### Using Docker Compose directly

If you cloned the repo:

```bash
# All services
docker compose up -d

# Just the essentials
docker compose up -d qdrant falkordb

# With PostgreSQL (production profile)
COMPOSE_PROFILES=production docker compose up -d
```

### Using native Ollama

If Ollama is already running natively (not in Docker), skip the `ollama` service:

```bash
re-memory setup -s qdrant -s falkordb   # no ollama flag

# Pull the recommended embedding model
ollama pull mxbai-embed-large
```

---

## Initialize

```bash
re-memory init
```

This creates:
- `~/.re-memory/` data directory
- `~/.re-memory/config.toml` configuration file
- `~/.re-memory/events.db` SQLite database
- `~/.re-memory/schemas/` directory for schema summaries
- Qdrant collection for vector embeddings (if available)
- FalkorDB graph indexes (if available)

---

## What Each Service Does

### Qdrant (port 6333)

**Vector similarity search.** When you `observe` something, the text is converted into a 1024-dimensional embedding. Qdrant stores these vectors and finds semantically similar memories during `recall`.

"What programming language does the user prefer?" will find "User prefers Python for scripting" even with zero word overlap.

### FalkorDB (port 6379)

**Knowledge graph.** Episodic memories get consolidated into structured triples:
```
(User) --[works_at]--> (OpenAI)
(User) --[prefers]--> (Python)
```
Triples have temporal scoping (valid_from, valid_to) to track when facts change.

### Ollama (port 11434)

**Local LLM inference.** Used for:
- Embedding generation (mxbai-embed-large, 1024 dimensions)
- Entity/topic extraction from input text
- Importance scoring (is this worth remembering?)
- Contradiction detection (does this conflict with existing memory?)
- Triple extraction during consolidation
- Schema summarization

### PostgreSQL (port 5432, optional)

**Production event store.** Replaces SQLite for high-concurrency deployments.

### What if services aren't running?

re-memory **degrades gracefully**:

| Service Down | What Happens |
|:---|:---|
| Qdrant unavailable | Falls back to SQLite text search (less accurate) |
| FalkorDB unavailable | Knowledge graph features disabled; episodic search still works |
| Ollama/LLM unavailable | SHA256 pseudo-embeddings, skips feature extraction and importance scoring |
| All containers down | Core observe/recall works via SQLite with fallback embeddings |

---

## Configuration

Config is loaded from (in priority order):
1. `./re_memory.toml` — project-local (highest priority)
2. `~/.re-memory/config.toml` — user-global (created by `re-memory init`)
3. Built-in defaults

```toml
[llm]
provider = "deepseek"         # ollama, openai, anthropic, deepseek
model = "deepseek-chat"       # use chat models, NOT reasoning models
base_url = "https://api.deepseek.com"
api_key = "sk-..."

[embedding]
provider = "ollama"           # ollama, openai
model = "mxbai-embed-large"   # recommended for entity discrimination
base_url = "http://localhost:11434"
dimensions = 1024

[storage]
qdrant_url = "http://localhost:6333"
falkordb_host = "localhost"
falkordb_port = 6379
sqlite_path = "~/.re-memory/events.db"
schema_dir = "~/.re-memory/schemas"

[consolidation]
interval_hours = 24           # daemon cycle frequency
decay_rate = 0.1              # Ebbinghaus decay factor (higher = faster forgetting)
confidence_threshold = 0.3    # below this, memories get pruned
max_episodic_age_days = 30    # max age before forced consolidation

[memory]
working_memory_slots = 7      # theta-gamma capacity (~7 items)
max_retrieval_tokens = 2000   # token budget per recall response
novelty_threshold = 0.3       # how different input must be to count as "new"
importance_threshold = 0.5    # below this, input may not be stored
```

### Provider examples

<details>
<summary><b>Ollama (fully local, no API keys)</b></summary>

```toml
[llm]
provider = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"

[embedding]
provider = "ollama"
model = "mxbai-embed-large"
base_url = "http://localhost:11434"
dimensions = 1024
```

</details>

<details>
<summary><b>OpenAI</b></summary>

```toml
[llm]
provider = "openai"
model = "gpt-4o-mini"
api_key = "sk-..."

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 768
api_key = "sk-..."
```

</details>

<details>
<summary><b>DeepSeek + Ollama embeddings (recommended for cost)</b></summary>

```toml
[llm]
provider = "deepseek"
model = "deepseek-chat"
api_key = "sk-..."

[embedding]
provider = "ollama"
model = "mxbai-embed-large"
dimensions = 1024
```

Note: use `deepseek-chat`, not `deepseek-reasoner`. The reasoner model spends 30-60s "thinking" per call — a ~12x slowdown for simple extraction tasks.

</details>

<details>
<summary><b>Anthropic + OpenAI embeddings</b></summary>

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
api_key = "sk-ant-..."

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 768
api_key = "sk-..."
```

Anthropic and DeepSeek don't provide embedding APIs, so pair with Ollama or OpenAI for embeddings.

</details>

---

## CLI Reference

### Setup & Lifecycle

```bash
re-memory setup -s qdrant -s falkordb  # Start selected infrastructure services
re-memory setup --all                   # Start all services
re-memory setup status                  # Check running services
re-memory setup stop                    # Stop all services
re-memory init                          # Initialize memory stores + config
re-memory purge --yes                   # Wipe all data (skip confirmation with --yes)
```

### Core Operations

```bash
# Store a memory
re-memory observe "User prefers dark mode in all applications"
re-memory observe "Meeting moved to Thursdays" --source "calendar-agent"

# Retrieve memories
re-memory recall "What are the user's UI preferences?"
re-memory recall "meetings" --limit 5 --max-tokens 500

# Manual consolidation (decay, prune, promote)
re-memory consolidate

# Explicitly forget
re-memory forget <memory-id>
```

### Introspection

```bash
re-memory status                      # System health + counts
re-memory inspect <memory-id>         # Full metadata for one memory
re-memory search "Python"             # Direct text search
re-memory history --limit 50          # Recent operations timeline
re-memory config                      # View current config
```

### Data Management

```bash
re-memory export ~/memory-backup.json  # Backup
re-memory import ~/memory-backup.json  # Restore
```

### Background Daemon

```bash
re-memory daemon start                 # Start consolidation daemon
re-memory daemon status                # Check if running
re-memory daemon stop                  # Stop daemon
```

### JSON Mode

Add `--json` (or `-j`) to any command for machine-readable output:

```bash
re-memory observe "User prefers vim keybindings" --json
```

```json
{
  "status": "encoded",
  "id": "a1b2c3d4-...",
  "verdict": "novel",
  "prediction_error": 1.0,
  "confidence": 0.5,
  "importance": 0.7,
  "sparse_code_bits": 64,
  "tags": ["preferences", "tools"],
  "timestamp": "2026-02-19T18:33:30+00:00"
}
```

```bash
re-memory recall "editor preferences" --json
```

```json
{
  "query": "editor preferences",
  "memories": [
    {
      "layer": "episodic",
      "id": "a1b2c3d4-...",
      "content": "User prefers vim keybindings",
      "confidence": 0.5,
      "importance": 0.7,
      "score": 0.82,
      "raw_similarity": 0.76,
      "access_count": 1,
      "created_at": "2026-02-19T18:33:30+00:00"
    }
  ],
  "total": 1,
  "token_budget_used": 7,
  "token_budget_max": 2000
}
```

---

## Understanding the Output

### Observe response fields

| Field | Meaning |
|:---|:---|
| `status` | `encoded` (stored) or `redundant` (already known, reinforced existing) |
| `verdict` | `novel` (new info), `redundant` (duplicate), `update` (partial change), `contradicts` (conflicts with existing) |
| `prediction_error` | 0.0 = identical to existing, 1.0 = completely new |
| `confidence` | Initial confidence score (0-1), decays over time without reinforcement |
| `importance` | How important this memory is (0-1), affects decay rate |
| `sparse_code_bits` | Number of active bits in the DG sparse code (always 64) |

### Recall response fields

| Field | Meaning |
|:---|:---|
| `layer` | Which memory layer: `schema`, `semantic`, or `episodic` |
| `score` | Combined relevance + recency score (0-1) |
| `raw_similarity` | Pure semantic similarity before time-decay adjustment |
| `confidence` | Current confidence (may have decayed since storage) |
| `access_count` | How many times this memory has been recalled |
| `token_budget_used` | Total tokens consumed by returned memories |

---

## Agent Integration

### Subprocess pattern

```python
import subprocess, json

def observe(text: str, source: str = "my-agent") -> dict:
    result = subprocess.run(
        ["re-memory", "observe", text, "--source", source, "--json"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def recall(query: str, max_tokens: int = 1000) -> list[dict]:
    result = subprocess.run(
        ["re-memory", "recall", query, "--max-tokens", str(max_tokens), "--json"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)["memories"]

# In your agent loop:
memories = recall("What does the user prefer?")
context = "\n".join(m["content"] for m in memories)
# Feed `context` into your LLM prompt...

# After the conversation:
observe("User said they prefer async Python over threads")
```

### Direct Python API

```python
from re_memory.engine import MemoryEngine

engine = MemoryEngine()
engine.init()

result = engine.observe("User prefers Python", source="my-agent")
memories = engine.recall("programming language")

for m in memories["memories"]:
    print(m["content"], m["score"])

engine.consolidate()
engine.purge()      # wipe all data
engine.close()
```

---

## Performance

Benchmarked with DeepSeek Chat + Ollama mxbai-embed-large + Qdrant + FalkorDB:

| Command | Avg Time | Notes |
|:---|---:|:---|
| **observe** | 4.0s | LLM calls (parallel: embed + features + importance) |
| **recall** | 143ms | Qdrant + SQLite, no LLM |
| **search** | 0.9ms | Pure SQLite |
| **consolidate** | 30s | Batch LLM calls (background job) |
| **forget** | 84ms | Cascade: SQLite + Qdrant + FalkorDB |
| **purge** | <100ms | Wipe all stores |
| **status** | 224ms | Health checks |
| **inspect** | 0.2ms | SQLite lookup |
| **history** | 2.9ms | SQLite query |
| **export/import** | 6-8ms | File I/O |

### Performance tips

- Use `deepseek-chat` not `deepseek-reasoner` (~12x faster for same quality)
- Use `mxbai-embed-large` (1024d) over `nomic-embed-text` (768d) for better entity discrimination
- Consolidation runs in background — don't wait for it interactively
- Recall and search are always fast (no LLM calls)

### Running the benchmark

```bash
python bench.py    # requires running infrastructure
```

---

## Running Tests

```bash
# All tests (68 total)
python -m pytest tests/ -v

# By category
python -m pytest tests/test_brain/ -v           # Brain components
python -m pytest tests/test_loops/ -v           # Processing loops
python -m pytest tests/test_integration.py -v   # End-to-end scenarios
python -m pytest tests/test_cli.py -v           # CLI tests
```

---

## Project Structure

```
re-memory/
├── src/re_memory/
│   ├── brain/                 # Brain components (EC, DG, CA3, CA1, PFC, BG, Amygdala, Neocortex)
│   ├── loops/                 # Processing loops (encoding, retrieval, consolidation, reconsolidation)
│   ├── memory/                # Memory layer interfaces (episodic, semantic, schema, working)
│   ├── providers/             # LLM/embedding providers (Ollama, OpenAI, Anthropic, DeepSeek)
│   ├── storage/               # Storage backends (SQLite, Qdrant, FalkorDB, files)
│   ├── engine.py              # Main orchestrator
│   ├── cli.py                 # Typer CLI (setup, observe, recall, purge, etc.)
│   └── config.py              # TOML configuration loader
├── rust/src/                  # Optional Rust accelerator (PyO3) — pattern separation, Hopfield, decay
├── tests/                     # 68 tests (pytest)
├── bench.py                   # Performance benchmark
├── docker-compose.yml         # Full infrastructure
└── docs/
    ├── Architecture.md        # Deep technical reference
    └── setup-and-usage.md     # This file
```
