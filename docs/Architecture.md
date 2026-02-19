# Architecture

re-memory is a standalone memory engine for AI agents that structurally mirrors the human brain's memory architecture. This document describes the system's internal design.

---

## Design Philosophy

The system maps real brain regions and their computational roles to software components — not as loose metaphors, but as faithful implementations of the underlying neuroscience:

- **Pattern separation** (Dentate Gyrus) ensures similar experiences get distinct fingerprints
- **Attractor dynamics** (CA3) enable full memory retrieval from partial cues
- **Prediction error** (CA1) gates what gets stored vs. ignored
- **Emotional tagging** (Amygdala) prioritizes important memories
- **Sleep replay** (Sharp-wave ripples) consolidates, prunes, and abstracts
- **Reconsolidation** allows recalled memories to be modified in context

The engine is agent-agnostic. Agents call `observe()` and `recall()` — they don't control how memory works internally.

---

## The 5-Layer Memory Stack

```
Layer 0: SENSORY BUFFER (Entorhinal Cortex)
  Raw input parsing, embedding generation, feature extraction
  TTL: single request

Layer 1: WORKING MEMORY (Prefrontal Cortex + Basal Ganglia)
  ~7-slot active context buffer with LRU eviction
  Gated read/write via policy
  TTL: one session

Layer 2: EPISODIC MEMORY (Hippocampus: DG -> CA3 -> CA1)
  Timestamped events with embeddings and sparse codes
  Pattern separation on write, pattern completion on read
  Novelty detection gates encoding
  TTL: days to weeks (provisional, awaiting consolidation)

Layer 3: SEMANTIC MEMORY (Knowledge Graph)
  Subject-predicate-object triples with temporal scoping
  Derived from episodic events via consolidation
  Active vs archived facts with confidence scores
  TTL: months to permanent

Layer 4: SCHEMA MEMORY (Abstractions)
  Human-readable category summaries as markdown files
  Continuously refined by consolidation
  Token-efficient retrieval target
  TTL: permanent
```

Each layer has different storage costs, retrieval speeds, and lifespans. Retrieval checks them top-down (cheapest first).

---

## The 3 Core Loops

### 1. Encoding (Write Path)

**Source**: `src/re_memory/loops/encoding.py`

```
Input text
  |
  v
Entorhinal Cortex: parse + embed
  |  - Generate 768-dim embedding (Ollama/OpenAI or SHA256 fallback)
  |  - Extract entities, topics, sentiment via LLM (optional)
  v
Dentate Gyrus: pattern separate
  |  - Sparse random projection: 768-dim -> 2048-dim -> top-64 bits
  |  - Produces a unique fingerprint even for similar inputs
  v
CA1: novelty detection
  |  - Search vector store for nearest memories (Qdrant or text fallback)
  |  - Compute prediction error: how different is this from what we know?
  |  - Verdict: NOVEL / REDUNDANT / UPDATE / CONTRADICTS
  v
Branch on verdict:
  |
  |-- REDUNDANT: reinforce existing memory (+0.05 confidence), skip write
  |
  |-- CONTRADICTS: lower old memory confidence, create new event
  |     (LLM verifies contradiction if available)
  |
  |-- NOVEL or UPDATE:
        |
        v
      Amygdala: importance scoring
        |  - LLM rates salience 0-1 (default 0.5 without LLM)
        v
      Create EpisodicEvent
        |  - Store in SQLite (episodic event store)
        |  - Store embedding in Qdrant (vector store)
        |  - Return: {id, verdict, prediction_error, confidence, importance}
```

**Key design choice**: Redundant input reinforces instead of duplicating. This means repeated observations of the same fact increase confidence rather than creating clutter.

### 2. Retrieval (Read Path)

**Source**: `src/re_memory/loops/retrieval.py`

```
Query text
  |
  v
Embed query (same provider as encoding)
  |
  v
Layer 4: Schema search (file_store.search)
  |  - Text search across markdown summary files
  |  - Cheapest retrieval — pre-summarized knowledge
  v
Layer 3: Knowledge Graph search (graph_store.search_triples)
  |  - FalkorDB Cypher queries for matching triples
  |  - Structured facts with temporal scoping
  v
Layer 2: Episodic retrieval
  |  - Vector similarity search (Qdrant, threshold 0.3)
  |  - CA3 pattern completion on stored embeddings
  |     (Modern Hopfield Network: softmax attention, beta=8.0)
  |  - Text search fallback if vector store unavailable
  v
Time-decay ranking
  |  - score = 0.7 * similarity + 0.3 * recency_decay
  |  - recency_decay = e^(-0.693 * hours / 168) (1-week half-life)
  |  - Rust-accelerated, Python fallback
  v
Token budget enforcement
  |  - Estimate tokens per memory (~4 chars/token)
  |  - Include top-scored memories until budget exhausted
  |  - Mark included memories as accessed (bumps access_count)
  v
Return ranked memories with metadata
```

**Key design choice**: Tiered retrieval (schema -> graph -> episodic) minimizes compute and token cost. Schema summaries are checked first because they're the most condensed representation.

### 3. Consolidation (Sleep Analogue)

**Source**: `src/re_memory/loops/consolidation.py`

```
Trigger: manual (re-memory consolidate) or daemon (APScheduler)
  |
  v
Step 1 - REPLAY
  |  Load all episodic events
  v
Step 2 - PROMOTE
  |  For unconsolidated events with confidence >= 0.4 and importance >= 0.3:
  |    - LLM extracts triples: (subject, predicate, object)
  |    - Store triples in FalkorDB with bi-temporal scoping
  |    - Detect conflicts: if new triple contradicts existing active triple,
  |      archive old triple (set valid_to), activate new
  |    - Mark event as consolidated
  v
Step 3 - ABSTRACT
  |  Group promoted triples by subject entity
  |  LLM generates/updates markdown schema summaries per topic
  v
Step 4 - DECAY
  |  For every event:
  |    - Compute decayed confidence using Ebbinghaus curve
  |    - Stability = f(access_count, importance, decay_rate)
  |    - More accesses + higher importance = slower decay
  |    - Update confidence if changed by > 0.01
  v
Step 5 - PRUNE
  |  Delete events below confidence_threshold (default 0.3)
  v
Step 6 - MERGE
  |  Find exact text duplicates (case-insensitive)
  |  Keep the one with higher confidence, delete the other
```

**Key design choice**: Consolidation is the only path from episodic to semantic memory. Raw events don't automatically become structured knowledge — they must survive long enough and be important enough to be promoted. This mirrors how the brain requires hippocampal replay during sleep to transfer memories to the cortex.

---

## Reconsolidation

**Source**: `src/re_memory/loops/reconsolidation.py`

When a memory is recalled, it becomes labile (modifiable). The system computes a prediction error between the recalled memory and the current context:

| Prediction Error | Action | Effect |
|---|---|---|
| PE < 0.2 | CONFIRM | Bump confidence +0.05 (strengthen) |
| 0.2 <= PE <= 0.7 | UPDATE | Labilize and update memory in place |
| PE > 0.7 | BIFURCATE | Create new episodic event, leave original unchanged |

This mirrors real reconsolidation: recalled memories can be strengthened, modified, or superseded depending on how much the current context differs from the original encoding context.

---

## Brain Components

### Entorhinal Cortex (`brain/entorhinal.py`)

Input gateway. Parses raw text into a structured representation:
- Generates embedding vector (768-dim default)
- Extracts entities, topics, sentiment, temporal references via LLM
- Falls back gracefully if LLM is unavailable (embedding-only mode)

### Dentate Gyrus (`brain/dentate_gyrus.py` + `rust/src/dentate_gyrus.rs`)

Pattern separation via sparse random projection:

1. Input: 768-dim dense embedding
2. Project through ternary random matrix (-1, 0, +1) into 2048-dim space
   - 2/3 of projection weights are zero (sparse matrix)
   - Matrix is deterministic (seeded RNG)
3. Winner-take-all: keep only top-64 activations
4. Output: sorted list of 64 active bit indices

This ensures even highly similar inputs produce distinct sparse codes, preventing catastrophic interference in storage.

### CA3 (`brain/ca3.py` + `rust/src/ca3.rs`)

Pattern completion via Modern Hopfield Network:

1. Compute scaled dot-product similarity between query and all stored patterns
2. Apply softmax with inverse temperature beta (default 8.0)
3. Return top-k (index, probability) tuples

Higher beta produces sharper retrieval (strongly favoring the best match). Lower beta distributes attention more evenly across similar memories.

Reference: Ramsauer et al., "Hopfield Networks is All You Need" (2020)

### CA1 (`brain/ca1.py`)

Novelty detection via prediction error:

- Compares new input embedding against nearest existing memories
- Prediction error = 1.0 - max_similarity
- Verdict thresholds (where `T` = novelty_threshold, default 0.3):
  - PE < T: REDUNDANT (very similar to existing memory)
  - T ≤ PE ≤ (1 - T): UPDATE (moderate difference, possible update)
  - PE > (1 - T): NOVEL (significantly different from all memories)
- LLM-based contradiction detection runs for both UPDATE and REDUNDANT
  verdicts (when PE ≥ 0.1) to catch semantically similar but factually
  contradictory statements (e.g., job changes)

### Amygdala (`brain/amygdala.py`)

Importance/salience scoring:
- Asks LLM: "On a scale of 0 to 1, how important is this to remember?"
- Returns float 0-1 (default 0.5 without LLM)
- `should_encode()` gate: rejects inputs below importance_threshold

### Prefrontal Cortex (`brain/prefrontal.py`)

Working memory buffer:
- OrderedDict-based LRU cache
- Default capacity: 7 slots (theta-gamma capacity from neuroscience)
- `add()` returns evicted item if at capacity
- `get()` touches item (moves to end of LRU order)
- `to_context_string()` serializes buffer for LLM context injection

### Basal Ganglia (`brain/basal_ganglia.py`)

Gating policy for working memory:
- Decides whether new items should ADMIT (enter WM), REJECT, or REPLACE an existing item
- Rule-based: relevance > 0.5 to admit; replaces lowest-relevance item if at capacity

### Neocortex (`brain/neocortex.py`)

Long-term knowledge coordinator:
- `extract_triples()`: LLM extracts (subject, predicate, object) from text, stores in FalkorDB
- `update_schema()`: LLM generates markdown summaries from groups of triples
- Conflict detection: compares new triples against existing active triples, archives contradicted ones

---

## Storage Layer

### SQLite Event Store (`storage/event_store.py`)

Primary episodic storage. Schema:

```
episodic_events:
  id            TEXT PRIMARY KEY
  text          TEXT
  embedding     BLOB (JSON-encoded float list)
  sparse_code   BLOB (JSON-encoded index list)
  confidence    REAL (0-1, decays over time)
  importance    REAL (0-1, set at encoding)
  tags          TEXT (JSON array)
  source        TEXT
  metadata      TEXT (JSON object)
  access_count  INTEGER (bumped on recall)
  synaptic_tag  REAL (TTL for consolidation eligibility)
  consolidated  INTEGER (0/1)
  created_at    TEXT (ISO 8601)
  last_accessed TEXT (ISO 8601)

memory_operations:
  id            INTEGER PRIMARY KEY
  operation     TEXT (observe, recall, consolidate, forget)
  memory_id     TEXT
  details       TEXT (JSON)
  timestamp     TEXT (ISO 8601)
```

Indexes on: `created_at`, `confidence`, `importance`, `consolidated`, `source`.

### Qdrant Vector Store (`storage/vector.py`)

Vector similarity search. Collection `re_memory_embeddings` with COSINE distance metric. Used for fast nearest-neighbor retrieval during both encoding (novelty detection) and retrieval (similarity search).

### FalkorDB Graph Store (`storage/graph.py`)

Knowledge graph with bi-temporal model:

```cypher
(Entity)-[RELATES_TO {
  predicate: "works_at",
  confidence: 0.9,
  valid_from: "2024-01-15",
  valid_to: null,           -- null = currently active
  ingested_at: "2024-01-16",
  active: true,
  source_event: "abc-123"
}]->(Entity)
```

When a new triple contradicts an existing active triple (same subject + predicate), the old one gets archived (`active=false`, `valid_to` set) and the new one becomes active.

### File Store (`storage/file_store.py`)

Schema summaries as markdown files in `~/.re-memory/schemas/`:

```
schemas/
  work.md        -- "User works at OpenAI on safety research"
  preferences.md -- "User prefers Python, dark mode, Neovim"
  projects.md    -- "Working on re-memory, a brain-anatomical engine"
```

Each file starts with a timestamp comment. Files are the cheapest retrieval target — plain text search, no database overhead.

---

## Rust Core

Performance-critical brain math compiled as a Python extension via PyO3 + maturin:

```
rust/src/
  lib.rs            -- PyO3 module entry, exports 12 functions
  dentate_gyrus.rs  -- pattern_separate, batch_pattern_separate
  ca3.rs            -- pattern_complete, batch_pattern_complete
  decay.rs          -- ebbinghaus_retention, compute_confidence_decay, time_decay_score
  sparse.rs         -- hamming_distance, jaccard_similarity, sparse_overlap
  similarity.rs     -- cosine_similarity, batch_cosine_similarity
```

Imported in Python as:
```python
from re_memory._core import pattern_separate, cosine_similarity, time_decay_score
```

Every Rust function has a pure-Python fallback using identical algorithms. If the Rust extension can't be imported (e.g., on a platform without a Rust toolchain), the system silently falls back to Python.

---

## Graceful Degradation

Every external dependency is optional:

| Dependency | If Available | If Unavailable |
|---|---|---|
| **Qdrant** | Fast vector similarity search for encoding and retrieval | Falls back to SQLite `LIKE` text search |
| **FalkorDB** | Knowledge graph queries, triple storage, conflict detection | Graph features disabled; episodic-only retrieval |
| **Ollama/OpenAI/Anthropic** | Real embeddings, feature extraction, importance scoring, triple extraction | SHA256-seeded pseudo-embeddings, default importance (0.5), skip LLM features |
| **Rust extension** | Fast math for DG, CA3, decay, similarity | Pure-Python fallback implementations |

The only hard dependency is **SQLite** (Python stdlib). Everything else degrades gracefully with try/except at every integration point.

---

## Decay Mathematics

Memory confidence decays according to the Ebbinghaus forgetting curve:

```
R = C * e^(-t / S)

Where:
  R = current retention (decayed confidence)
  C = base confidence at last access
  t = hours since last access
  S = stability factor
```

Stability is computed from access patterns and importance:

```
S = (1 + ln(1 + access_count)) * (0.5 + importance) / decay_rate
```

This means:
- **More accesses** = higher stability = slower decay (repeated recall strengthens)
- **Higher importance** = higher stability = slower decay (salient memories persist)
- **Higher decay_rate** = lower stability = faster forgetting (configurable global knob)

For retrieval ranking, time-decay is combined with relevance:

```
score = (1 - w) * similarity + w * recency_decay
recency_decay = e^(-0.693 * hours / half_life)

Default: w=0.3, half_life=168 hours (1 week)
```

---

## Data Flow Summary

```
observe("User works at OpenAI")
  |
  v
[Encoding Loop]
  EC: embed -> DG: sparse code -> CA1: NOVEL -> Amygdala: 0.7
  |
  +-> SQLite: episodic event (confidence=0.5, importance=0.7)
  +-> Qdrant: embedding vector with metadata
  |
  v
consolidate()
  |
  v
[Consolidation Loop]
  Replay events -> Promote (confidence>=0.4, importance>=0.3)
  |
  +-> FalkorDB: (User)-[works_at]->(OpenAI)
  +-> schemas/user.md: "User works at OpenAI"
  |
  Decay unreinforced -> Prune below 0.3 -> Merge duplicates
  |
  v
recall("Where does the user work?")
  |
  v
[Retrieval Loop]
  Schema search: "User works at OpenAI" (Layer 4)
  KG search: (User)-[works_at]->(OpenAI) (Layer 3)
  Vector search: episodic events (Layer 2)
  |
  Rank by relevance + recency -> Enforce token budget
  |
  v
Return: [{layer: "schema", content: "User works at OpenAI", score: 1.0}, ...]
```

---

## Configuration

All behavior is tunable via `~/.re-memory/config.toml`:

| Section | Key Parameters |
|---|---|
| `[llm]` | provider, model, api_key |
| `[embedding]` | provider, model, dimensions (768 default) |
| `[storage]` | qdrant_url, falkordb_host/port, sqlite_path, schema_dir |
| `[consolidation]` | interval_hours (24), decay_rate (0.1), confidence_threshold (0.3), max_episodic_age_days (30) |
| `[memory]` | working_memory_slots (7), max_retrieval_tokens (2000), novelty_threshold (0.3), importance_threshold (0.5) |

See `docs/setup-and-usage.md` for full configuration reference.
