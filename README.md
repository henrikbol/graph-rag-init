# GraphRAG MVP

A prototype for experimenting with [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — build a knowledge graph from text documents, visualize it interactively, and query it with natural language.

Uses **Claude** (via LiteLLM) for graph extraction and summarization, and **OpenAI** for text embeddings.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An Anthropic API key (for Claude)
- An OpenAI API key (for embeddings)

## Setup

```bash
# Install dependencies
uv sync
```

Add your keys to `.env`:

```
GRAPHRAG_API_KEY=sk-...        # OpenAI key (embeddings)
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic key (Claude)
```

## Usage

### 1. Index documents

Place `.txt` files in `input/` (sample Apollo Program documents are included), then run:

```bash
uv run python main.py index
```

This runs the full GraphRAG pipeline: chunking, entity/relationship extraction, community detection, and report generation. Output is written as parquet files to `output/`.

### 2. Visualize the knowledge graph

```bash
uv run python main.py visualize
open graph.html
```

Generates an interactive `graph.html` file with:
- Nodes colored by entity type (red=person, blue=organization, green=geo, yellow=event)
- Node size scaled by degree centrality
- Hover tooltips with entity/relationship descriptions
- Physics-based force layout

### 3. Query the knowledge graph

```bash
# Global search (searches across community reports)
uv run python main.py query "What were the key technological achievements of Apollo?"

# Local search (searches specific entity neighborhoods)
uv run python main.py query --local "Who was Wernher von Braun?"
```

## Project Structure

```
.
├── main.py              # CLI entry point (index / visualize / query)
├── settings.yaml        # GraphRAG configuration
├── .env                 # API keys (not committed)
├── input/               # Source text documents
├── output/              # Indexed parquet files + vector store
├── prompts/             # LLM prompt templates
├── cache/               # LLM response cache
└── logs/                # Pipeline run logs
```

## Configuration

All GraphRAG settings are in `settings.yaml`. Key options:

- **Completion model**: `anthropic/claude-sonnet-4-20250514` via LiteLLM
- **Embedding model**: `text-embedding-3-small` via OpenAI
- **Entity types**: organization, person, geo, event
- **Chunk size**: 1200 tokens with 100 token overlap

See the [GraphRAG config docs](https://microsoft.github.io/graphrag/config/yaml/) for all options.
