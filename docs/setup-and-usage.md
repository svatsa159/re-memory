# Technical Setup & Usage Guide

## Prerequisites

- Python 3.11+
- Rust toolchain (for building the native extension)
- Docker (for infrastructure services)

## Installation

### 1. Clone and install

```bash
git clone <repo-url>
cd re-memory

# Create virtual environment and install (builds Rust extension via maturin)
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This compiles the Rust core (pattern separation, Hopfield network, decay math) into a Python extension module automatically via maturin.

### 2. Start infrastructure (Docker)

re-memory uses three external services, all containerized:

```bash
# Development mode (lighter — no PostgreSQL)
docker compose -f docker-compose.dev.yml up -d

# Production mode (includes PostgreSQL)
docker compose up -d
```

### 3. Initialize

```bash
re-memory init
```

This creates:
- `~/.re-memory/` data directory
- `~/.re-memory/config.toml` configuration file
- `~/.re-memory/events.db` SQLite database
- `~/.re-memory/schemas/` directory for schema summaries
- Qdrant collection for vector embeddings
- FalkorDB graph indexes

---

## What Are the Docker Containers For?

re-memory uses a multi-store architecture. Each container serves a specific memory layer:

### Qdrant (port 6333)

**Purpose:** Vector similarity search for memory retrieval.

When you `observe` something, the text is converted into a numerical embedding (a 768-dimensional vector). Qdrant stores these vectors and enables fast nearest-neighbor search — so when you `recall` something, it finds the most semantically similar memories, not just keyword matches.

"What programming language does the user prefer?" will find "User prefers Python for scripting" even though no words overlap.

**Used by:** Encoding loop (store embeddings), Retrieval loop (similarity search)

### FalkorDB (port 6379)

**Purpose:** Knowledge graph for structured facts.

Episodic memories ("User said they work at OpenAI") get consolidated into structured triples:
```
(User) --[works_at]--> (OpenAI)
(User) --[prefers]--> (Python)
```

FalkorDB stores these as a graph with temporal scoping (valid_from, valid_to) so re-memory can track when facts changed. If the user switches jobs, the old triple gets archived and the new one becomes active.

**Used by:** Consolidation loop (store triples), Retrieval loop (graph queries), Neocortex (conflict detection)

### Ollama (port 11434)

**Purpose:** Local LLM and embedding inference.

When an LLM provider is configured, re-memory uses it for:
- **Embedding generation** — Converting text to vectors (nomic-embed-text)
- **Feature extraction** — Identifying entities, topics, sentiment from input
- **Importance scoring** — Deciding if something is worth remembering
- **Triple extraction** — Pulling structured facts from text during consolidation
- **Schema generation** — Summarizing groups of facts into readable summaries

Ollama runs these models locally with no API keys needed. You can also use OpenAI or Anthropic APIs instead.

**Used by:** Entorhinal cortex, Amygdala, Neocortex, all LLM-dependent features

### PostgreSQL (port 5432, production only)

**Purpose:** Production-grade event store replacing SQLite.

In development, episodic events are stored in SQLite (`~/.re-memory/events.db`). For production deployments with high concurrency, PostgreSQL provides better write throughput and durability. Only started with the `production` Docker profile.

### What if containers aren't running?

re-memory **degrades gracefully**. Every external dependency is optional:

| Service Down | What Happens |
|---|---|
| Qdrant unavailable | Falls back to text search in SQLite (less accurate but functional) |
| FalkorDB unavailable | Knowledge graph features disabled; episodic search still works |
| Ollama unavailable | Uses hash-based pseudo-embeddings and skips LLM features (importance scoring, feature extraction) |
| All containers down | Core observe/recall still works via SQLite text search with fallback embeddings |

---

## Configuration

Config file: `~/.re-memory/config.toml` (created by `re-memory init`)

```toml
[llm]
provider = "ollama"          # ollama, openai, anthropic
model = "llama3.2"
base_url = "http://localhost:11434"
api_key = ""                 # required for openai/anthropic

[embedding]
provider = "ollama"
model = "nomic-embed-text"
base_url = "http://localhost:11434"
dimensions = 768

[storage]
qdrant_url = "http://localhost:6333"
falkordb_host = "localhost"
falkordb_port = 6379
sqlite_path = "~/.re-memory/events.db"
schema_dir = "~/.re-memory/schemas"

[consolidation]
interval_hours = 24          # daemon cycle frequency
decay_rate = 0.1             # higher = faster forgetting
confidence_threshold = 0.3   # below this, memories get pruned
max_episodic_age_days = 30

[memory]
working_memory_slots = 7     # active context capacity
max_retrieval_tokens = 2000  # token budget per recall
novelty_threshold = 0.3      # how different input must be to count as "new"
importance_threshold = 0.5   # below this, input might not be stored
```

### Using OpenAI instead of Ollama

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

### Using Anthropic for LLM (with OpenAI for embeddings)

Anthropic doesn't provide an embedding API, so use a split config:

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
api_key = "sk-ant-..."

[embedding]
provider = "openai"
model = "text-embedding-3-small"
api_key = "sk-..."
```

### Using DeepSeek for LLM (with Ollama or OpenAI for embeddings)

DeepSeek uses an OpenAI-compatible API. Like Anthropic, it doesn't provide embeddings, so pair it with another embedding provider:

```toml
[llm]
provider = "deepseek"
model = "deepseek-reasoner"
api_key = "sk-..."

[embedding]
provider = "ollama"
model = "nomic-embed-text"
base_url = "http://localhost:11434"
dimensions = 768
```

---

## Runtime Usage

### CLI Commands

#### Core operations (what agents call)

```bash
# Store a memory
re-memory observe "User prefers dark mode in all applications"
re-memory observe "Meeting with design team moved to Thursdays" --source "calendar-agent"

# Retrieve memories
re-memory recall "What are the user's UI preferences?"
re-memory recall "meetings" --limit 5 --max-tokens 500

# Manual consolidation (decay, prune, promote)
re-memory consolidate

# Explicitly forget
re-memory forget <memory-id>
```

#### Introspection

```bash
# System health
re-memory status

# View specific memory with full metadata
re-memory inspect <memory-id>

# Direct text search across all layers
re-memory search "Python"

# Recent operations timeline
re-memory history --limit 50

# View current config
re-memory config
```

#### Data management

```bash
# Backup
re-memory export ~/memory-backup.json

# Restore
re-memory import ~/memory-backup.json
```

#### Background consolidation daemon

```bash
# Start daemon (runs consolidation every N hours per config)
re-memory daemon start

# Check if running
re-memory daemon status

# Stop
re-memory daemon stop
```

### JSON Mode (for agent integration)

Add `--json` (or `-j`) before any command for machine-readable output:

```bash
re-memory --json observe "User prefers vim keybindings"
```

```json
{
  "status": "encoded",
  "id": "a1b2c3d4-...",
  "verdict": "novel",
  "prediction_error": 1.0,
  "confidence": 0.5,
  "importance": 0.5,
  "sparse_code_bits": 64,
  "tags": [],
  "timestamp": "2026-02-17T18:33:30+00:00"
}
```

```bash
re-memory --json recall "editor preferences"
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
      "importance": 0.5,
      "score": 0.82,
      "raw_similarity": 0.76,
      "access_count": 1,
      "created_at": "2026-02-17T18:33:30+00:00"
    }
  ],
  "total": 1,
  "token_budget_used": 7,
  "token_budget_max": 2000
}
```

### Integrating With Your Agent

A typical agent integration pattern:

```python
import subprocess, json

def observe(text: str, source: str = "my-agent") -> dict:
    result = subprocess.run(
        ["re-memory", "--json", "observe", text, "--source", source],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def recall(query: str, max_tokens: int = 1000) -> list[dict]:
    result = subprocess.run(
        ["re-memory", "--json", "recall", query, "--max-tokens", str(max_tokens)],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return data["memories"]

# In your agent loop:
memories = recall("What does the user prefer?")
context = "\n".join(m["content"] for m in memories)
# Feed `context` into your LLM prompt...

# After the conversation:
observe("User said they prefer async Python over threads")
```

Or use the Python API directly:

```python
from re_memory.engine import MemoryEngine

engine = MemoryEngine()
engine.init()

# Store
result = engine.observe("User prefers Python", source="my-agent")

# Retrieve
memories = engine.recall("programming language")
for m in memories["memories"]:
    print(m["content"], m["score"])

# Consolidate
engine.consolidate()

engine.close()
```

---

## Understanding the Output

### Observe response fields

| Field | Meaning |
|---|---|
| `status` | `encoded` (stored) or `redundant` (already known, reinforced existing) |
| `verdict` | `novel` (new info), `redundant` (duplicate), `update` (partial change), `contradicts` (conflicts with existing) |
| `prediction_error` | 0.0 = identical to existing, 1.0 = completely new |
| `confidence` | Initial confidence score (0-1), decays over time without reinforcement |
| `importance` | How important this memory is (0-1), affects decay rate |
| `sparse_code_bits` | Number of active bits in the DG sparse code (always 64) |

### Recall response fields

| Field | Meaning |
|---|---|
| `layer` | Which memory layer: `schema`, `semantic`, or `episodic` |
| `score` | Combined relevance + recency score (0-1) |
| `raw_similarity` | Pure semantic similarity before time-decay adjustment |
| `confidence` | Current confidence (may have decayed since storage) |
| `access_count` | How many times this memory has been recalled |
| `token_budget_used` | Total tokens consumed by returned memories |

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test category
python -m pytest tests/test_brain/ -v       # Brain components
python -m pytest tests/test_loops/ -v       # Processing loops
python -m pytest tests/test_memory/ -v      # Storage backends
python -m pytest tests/test_integration.py -v  # End-to-end scenarios
```

## Project Structure

```
re-memory/
├── rust/src/              # Rust core (PyO3) — pattern separation, Hopfield net, decay
├── src/re_memory/
│   ├── brain/             # Brain components (EC, DG, CA3, CA1, PFC, BG, Amygdala, Neocortex)
│   ├── memory/            # Memory layer interfaces
│   ├── loops/             # Processing loops (encoding, retrieval, consolidation, reconsolidation)
│   ├── providers/         # LLM/embedding providers (Ollama, OpenAI, Anthropic)
│   ├── storage/           # Storage backends (SQLite, Qdrant, FalkorDB, files)
│   ├── cli.py             # Typer CLI
│   ├── engine.py          # Main orchestrator
│   └── config.py          # TOML configuration
├── tests/                 # 68 tests (pytest)
├── docker-compose.yml     # Full infrastructure
└── docker-compose.dev.yml # Lightweight dev setup
```
