<div align="center">

# re-memory

**A brain-anatomical memory engine for AI agents**

*Not a metaphor — actual neuroscience mapped to software.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-core-dea584?style=flat-square&logo=rust&logoColor=white)](https://rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-68_passing-brightgreen?style=flat-square)]()

---

Most AI agents forget everything between conversations.<br>
The solutions that exist — vector databases, RAG pipelines — are flat.<br>
They store and retrieve, but don't **process** memories the way brains do.

**re-memory** implements the real computational primitives from neuroscience.

</div>

---

## How the Brain Maps to Code

<table>
<tr>
<td width="50%">

### Brain Region Mapping

| Brain Region | Computational Role |
|:---|:---|
| **Entorhinal Cortex** | Input gateway — parse + embed |
| **Dentate Gyrus** | Pattern separation — sparse codes |
| **CA3** | Pattern completion — Hopfield recall |
| **CA1** | Novelty detection — prediction error |
| **Prefrontal Cortex** | Working memory — ~7 slot buffer |
| **Basal Ganglia** | Gating — what enters memory |
| **Amygdala** | Importance — salience scoring |
| **Neocortex** | Long-term — knowledge graph |
| **Sharp-wave Ripples** | Sleep — consolidation daemon |

</td>
<td width="50%">

### What This Gets You

- **Pattern separation** — similar inputs get distinct fingerprints, no interference
- **Pattern completion** — retrieve full memories from partial cues
- **Novelty gating** — only store what's genuinely new
- **Importance scoring** — prioritize what matters
- **Natural forgetting** — Ebbinghaus decay unless reinforced
- **Sleep consolidation** — promotes, abstracts, and prunes
- **Reconsolidation** — recalled memories update in context
- **Contradiction handling** — old facts archived, new ones stored

</td>
</tr>
</table>

---

## The 5-Layer Memory Stack

```
                              +--------------------------+
                              |   Layer 4: SCHEMA        |  Permanent abstractions
                              |   Category summaries     |  "User prefers Python"
                              +--------------------------+
                         +-------------------------------+
                         |   Layer 3: SEMANTIC            |  Months+
                         |   Knowledge graph triples      |  (User)-[prefers]->(Python)
                         +-------------------------------+
                    +------------------------------------+
                    |   Layer 2: EPISODIC                  |  Days to weeks
                    |   Timestamped events + embeddings    |  "User said they like Python"
                    +------------------------------------+
               +-----------------------------------------+
               |   Layer 1: WORKING MEMORY                |  One session
               |   ~7-slot active context buffer          |  Current conversation
               +-----------------------------------------+
          +----------------------------------------------+
          |   Layer 0: SENSORY BUFFER                      |  Single request
          |   Raw input parsing + embedding                |  Incoming text
          +----------------------------------------------+
```

Retrieval checks layers **top-down** (cheapest first). Consolidation promotes memories **bottom-up** (episodic -> semantic -> schema).

---

## Three Core Loops

### Write (Encoding)

```
Input ──> Entorhinal Cortex ──> Dentate Gyrus ──> CA1 ──> Amygdala ──> Store
          parse + embed         sparse fingerprint  new?    important?   SQLite + Qdrant
```

> Redundant input **reinforces** existing memories instead of duplicating. Contradictions are detected — the old fact gets archived, the new one takes over.

### Read (Retrieval)

```
Query ──> Schema Search ──> Knowledge Graph ──> Episodic Search ──> CA3 ──> Rank + Budget
          Layer 4 (fast)    Layer 3 (precise)   Layer 2 (ground     Hopfield  Time-decay
                                                 truth)              recall    scoring
```

> Results are ranked by `0.7 * relevance + 0.3 * recency`. Token budget enforcement prevents context overflow.

### Consolidation (Sleep)

```
Replay ──> Promote ──> Abstract ──> Decay ──> Prune ──> Merge
events     to graph    to schemas   Ebbinghaus below     dedup
                                    curve      threshold
```

> Runs as a background daemon or on-demand. Unreinforced memories fade. Important memories get promoted to structured knowledge.

---

## Quick Start

```bash
# Clone and install (builds Rust extension automatically)
git clone https://github.com/your-org/re-memory.git
cd re-memory
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Start infrastructure (optional — works without Docker too)
docker compose -f docker-compose.dev.yml up -d

# Initialize
re-memory init
```

```bash
# Store memories
re-memory observe "User prefers Python for scripting"
re-memory observe "User works at OpenAI on safety research"
re-memory observe "User's favorite editor is Neovim"

# Recall
re-memory recall "What programming language?"
re-memory recall "Where does the user work?"

# Consolidate (decay, prune, promote to knowledge graph)
re-memory consolidate

# Forget explicitly
re-memory forget <memory-id>
```

<details>
<summary><b>All CLI commands</b></summary>

```bash
# Core operations
re-memory observe <text>              # Store a memory
re-memory recall <query>              # Retrieve memories
re-memory consolidate                 # Run consolidation cycle
re-memory forget <memory-id>          # Explicit forgetting

# Introspection
re-memory status                      # System health
re-memory inspect <memory-id>         # View memory with full metadata
re-memory search <query>              # Direct text search
re-memory history --limit 50          # Recent operations timeline
re-memory config                      # View current config

# Data management
re-memory export backup.json          # Backup all memories
re-memory import backup.json          # Restore from backup

# Background daemon
re-memory daemon start                # Start consolidation daemon
re-memory daemon stop                 # Stop daemon
re-memory daemon status               # Check daemon state

# JSON mode (for agent integration)
re-memory --json observe "..."        # Machine-readable output
re-memory --json recall "..."         # Structured JSON responses
```

</details>

---

## Agent Integration

re-memory is **agent-agnostic**. Any agent can use it as an external memory service.

### CLI with JSON mode

```bash
re-memory --json observe "User prefers async Python over threads"
```

```json
{
  "status": "encoded",
  "id": "a1b2c3d4-...",
  "verdict": "novel",
  "prediction_error": 1.0,
  "confidence": 0.5,
  "importance": 0.5,
  "sparse_code_bits": 64
}
```

### Python API

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

### Subprocess integration

```python
import subprocess, json

def recall(query: str) -> list[dict]:
    result = subprocess.run(
        ["re-memory", "--json", "recall", query],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)["memories"]

# Feed into your LLM prompt
memories = recall("What does the user prefer?")
context = "\n".join(m["content"] for m in memories)
```

---

## Graceful Degradation

Every external dependency is optional. re-memory works with just Python and SQLite.

| Dependency | If Running | If Unavailable |
|:---|:---|:---|
| **Qdrant** | Fast vector similarity search | Falls back to SQLite text search |
| **FalkorDB** | Knowledge graph + structured facts | Graph features disabled, episodic-only |
| **Ollama / OpenAI / Anthropic** | Real embeddings + LLM features | SHA256 pseudo-embeddings, default scoring |
| **Rust extension** | Fast math (DG, CA3, decay) | Pure-Python fallback (same algorithms) |
| **All containers down** | --- | Core observe/recall still works via SQLite |

---

## Tech Stack

| Component | Technology | Purpose |
|:---|:---|:---|
| **Core** | Python + Rust (PyO3/maturin) | Python orchestration, Rust hot-path math |
| **Event Store** | SQLite / PostgreSQL | Episodic memory storage |
| **Vector Store** | Qdrant | Embedding similarity search |
| **Knowledge Graph** | FalkorDB | Structured facts with temporal scoping |
| **LLM** | Ollama, OpenAI, Anthropic, DeepSeek | Embeddings, feature extraction, scoring |
| **CLI** | Typer + Rich | User interface |
| **Background Jobs** | APScheduler | Consolidation daemon |

---

## Configuration

Config lives at `~/.re-memory/config.toml`:

```toml
[llm]
provider = "ollama"       # ollama, openai, anthropic, deepseek
model = "llama3.2"

[embedding]
provider = "ollama"       # ollama, openai
model = "nomic-embed-text"
dimensions = 768

[consolidation]
decay_rate = 0.1          # higher = faster forgetting
confidence_threshold = 0.3
interval_hours = 24

[memory]
working_memory_slots = 7
max_retrieval_tokens = 2000
novelty_threshold = 0.3
importance_threshold = 0.5
```

<details>
<summary><b>Using OpenAI</b></summary>

```toml
[llm]
provider = "openai"
model = "gpt-4o-mini"
api_key = "sk-..."

[embedding]
provider = "openai"
model = "text-embedding-3-small"
api_key = "sk-..."
```

</details>

<details>
<summary><b>Using Anthropic + OpenAI embeddings</b></summary>

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

</details>

<details>
<summary><b>Using DeepSeek</b></summary>

```toml
[llm]
provider = "deepseek"
model = "deepseek-reasoner"
api_key = "sk-..."

[embedding]
provider = "ollama"
model = "nomic-embed-text"
```

</details>

---

## Project Structure

```
re-memory/
├── rust/src/                  # Rust core (PyO3) — pattern separation, Hopfield, decay
├── src/re_memory/
│   ├── brain/                 # Brain components (EC, DG, CA3, CA1, PFC, BG, Amygdala, Neocortex)
│   ├── loops/                 # Processing loops (encoding, retrieval, consolidation, reconsolidation)
│   ├── memory/                # Memory layer interfaces (episodic, semantic, schema, working)
│   ├── providers/             # LLM/embedding providers (Ollama, OpenAI, Anthropic, DeepSeek)
│   ├── storage/               # Storage backends (SQLite, Qdrant, FalkorDB, files)
│   ├── engine.py              # Main orchestrator
│   └── cli.py                 # Typer CLI
├── tests/                     # 68 tests (pytest)
├── docker-compose.yml         # Full infrastructure
└── docs/
    ├── Architecture.md        # Deep technical reference
    └── setup-and-usage.md     # Setup guide + runtime usage
```

---

## Documentation

- **[Architecture](docs/Architecture.md)** — Brain mappings, data flow, decay math, storage schemas
- **[Setup & Usage](docs/setup-and-usage.md)** — Installation, Docker, configuration, CLI reference, Python API

---

## License

MIT
