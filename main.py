"""GraphRAG MVP — index documents, visualize the knowledge graph, and query it."""

import asyncio
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from graphrag.api import build_index, global_search, local_search
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config

ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "output"

# Entity type → color mapping for visualization
ENTITY_COLORS: dict[str, str] = {
    "PERSON": "#e74c3c",
    "ORGANIZATION": "#3498db",
    "GEO": "#2ecc71",
    "EVENT": "#f39c12",
}
DEFAULT_COLOR = "#95a5a6"


def load_config_from_root() -> object:
    """Load GraphRAG config from the project root."""
    return load_config(ROOT_DIR)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


async def run_index() -> None:
    """Run the GraphRAG indexing pipeline."""
    config = load_config_from_root()
    print("Starting indexing pipeline…")
    results = await build_index(
        config=config,
        method=IndexingMethod.Standard,
        verbose=True,
    )
    for result in results:
        print(f"  Workflow: {result.workflow}")
        if result.error:
            print(f"    ERROR: {result.error}")
    print("Indexing complete. Output written to output/")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def run_visualize() -> None:
    """Build an interactive HTML graph from the indexed knowledge graph."""
    entities_path = OUTPUT_DIR / "entities.parquet"
    relationships_path = OUTPUT_DIR / "relationships.parquet"

    if not entities_path.exists() or not relationships_path.exists():
        print("Error: parquet files not found in output/. Run 'index' first.")
        sys.exit(1)

    entities_df = pd.read_parquet(entities_path)
    relationships_df = pd.read_parquet(relationships_path)

    print(f"Loaded {len(entities_df)} entities, {len(relationships_df)} relationships")

    # Build NetworkX graph
    graph = nx.DiGraph()

    for _, row in entities_df.iterrows():
        entity_type = str(row.get("type", "")).upper()
        degree = int(row.get("degree", 1))
        graph.add_node(
            row["title"],
            label=row["title"],
            title=f"[{entity_type}] {row.get('description', '')}",
            color=ENTITY_COLORS.get(entity_type, DEFAULT_COLOR),
            size=10 + degree * 3,
            entity_type=entity_type,
        )

    for _, row in relationships_df.iterrows():
        weight = float(row.get("weight", 1.0))
        graph.add_edge(
            row["source"],
            row["target"],
            title=str(row.get("description", "")),
            value=weight,
        )

    # Create PyVis visualization
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )
    net.from_nx(graph)
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.02
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    output_file = ROOT_DIR / "graph.html"
    net.save_graph(str(output_file))
    print(f"Graph saved to {output_file}")
    print(f"Open it in your browser: file://{output_file.resolve()}")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


async def run_query(query: str, method: str = "global") -> None:
    """Query the knowledge graph."""
    config = load_config_from_root()

    entities_df = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
    communities_df = pd.read_parquet(OUTPUT_DIR / "communities.parquet")
    reports_df = pd.read_parquet(OUTPUT_DIR / "community_reports.parquet")

    if method == "global":
        print(f"Running global search: {query!r}\n")
        response, _context = await global_search(
            config=config,
            entities=entities_df,
            communities=communities_df,
            community_reports=reports_df,
            community_level=None,
            dynamic_community_selection=True,
            response_type="multiple paragraphs",
            query=query,
        )
    elif method == "local":
        relationships_df = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
        text_units_df = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")

        print(f"Running local search: {query!r}\n")
        response, _context = await local_search(
            config=config,
            entities=entities_df,
            communities=communities_df,
            community_reports=reports_df,
            text_units=text_units_df,
            relationships=relationships_df,
            covariates=None,
            community_level=1,
            response_type="multiple paragraphs",
            query=query,
        )
    else:
        print(f"Unknown search method: {method}")
        sys.exit(1)

    print(response)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_usage() -> None:
    """Print usage information."""
    print(
        "Usage:\n"
        "  uv run python main.py index                          — Build the knowledge graph\n"
        "  uv run python main.py visualize                      — Generate interactive graph.html\n"
        '  uv run python main.py query "your question"          — Global search\n'
        '  uv run python main.py query --local "your question"  — Local search'
    )


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if not args:
        print_usage()
        sys.exit(0)

    command = args[0]

    if command == "index":
        asyncio.run(run_index())

    elif command == "visualize":
        run_visualize()

    elif command == "query":
        method = "global"
        query_args = args[1:]

        if query_args and query_args[0] == "--local":
            method = "local"
            query_args = query_args[1:]

        if not query_args:
            print("Error: provide a query string.")
            sys.exit(1)

        query = " ".join(query_args)
        asyncio.run(run_query(query, method))

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
