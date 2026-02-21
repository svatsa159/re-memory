"""Dataset downloader for benchmark datasets.

Downloads and caches datasets from HuggingFace Hub and GitHub repositories.
All datasets are stored under benchmarks/datasets/cache/ and are .gitignored.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"

# ── Dataset registry ─────────────────────────────────────────────────────

DATASETS = {
    "memory_agent_bench": {
        "source": "huggingface",
        "repo": "ai-hyz/MemoryAgentBench",
        "description": "MemoryAgentBench (ICLR 2026) — test-time learning, retrieval, consolidation, forgetting",
    },
    "locomo": {
        "source": "github",
        "repo": "snap-research/locomo",
        "description": "LoCoMo — 300-turn multi-session conversations, multi-hop reasoning",
    },
    "longmemeval": {
        "source": "github",
        "repo": "xiaowu0162/LongMemEval",
        "description": "LongMemEval (ICLR 2025) — temporal reasoning, knowledge updates, abstention",
    },
}


def ensure_cache_dir():
    """Create cache directory and .gitignore it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = CACHE_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")


def download_huggingface(dataset_name: str, repo: str) -> Path:
    """Download a dataset from HuggingFace using the datasets library."""
    ensure_cache_dir()
    dest = CACHE_DIR / dataset_name

    if dest.exists() and any(dest.iterdir()):
        print(f"  [cached] {dataset_name} already at {dest}")
        return dest

    print(f"  Downloading {repo} from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, cache_dir=str(dest))

        # Save splits as JSON for offline use
        for split_name in ds:
            split_path = dest / f"{split_name}.json"
            ds[split_name].to_json(str(split_path))
            print(f"    Saved {split_name} → {split_path.name} ({len(ds[split_name])} samples)")

        return dest
    except ImportError:
        print("  ERROR: `datasets` library not installed. Run: pip install datasets")
        raise
    except Exception as e:
        print(f"  ERROR downloading {repo}: {e}")
        raise


def download_github(dataset_name: str, repo: str) -> Path:
    """Clone a GitHub repository (shallow clone)."""
    ensure_cache_dir()
    dest = CACHE_DIR / dataset_name

    if dest.exists() and any(dest.iterdir()):
        print(f"  [cached] {dataset_name} already at {dest}")
        return dest

    url = f"https://github.com/{repo}.git"
    print(f"  Cloning {url} (shallow)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"    Cloned to {dest}")
        return dest
    except subprocess.CalledProcessError as e:
        print(f"  ERROR cloning {repo}: {e.stderr}")
        raise


def download_dataset(name: str) -> Path:
    """Download a dataset by name.

    Args:
        name: Key from DATASETS registry.

    Returns:
        Path to the downloaded dataset directory.
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    info = DATASETS[name]
    if info["source"] == "huggingface":
        return download_huggingface(name, info["repo"])
    elif info["source"] == "github":
        return download_github(name, info["repo"])
    else:
        raise ValueError(f"Unknown source type: {info['source']}")


def download_all() -> dict[str, Path]:
    """Download all registered datasets."""
    results = {}
    for name in DATASETS:
        print(f"\n{'─'*60}")
        print(f"Dataset: {name}")
        print(f"  {DATASETS[name]['description']}")
        try:
            results[name] = download_dataset(name)
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = None
    return results


def list_datasets():
    """Print available datasets and their cache status."""
    ensure_cache_dir()
    for name, info in DATASETS.items():
        cached = (CACHE_DIR / name).exists()
        status = "[cached]" if cached else "[not downloaded]"
        print(f"  {name:<25} {status:<20} {info['description']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_datasets()
    elif len(sys.argv) > 1:
        download_dataset(sys.argv[1])
    else:
        download_all()
