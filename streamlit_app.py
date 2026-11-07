import json
from pathlib import Path
from typing import Callable, Iterator, Protocol

import streamlit as st

from core_rag.agents_graph import CRITIC_LLM_MODEL, LLM_MODEL
from core_rag.vector_store import (
    CHROMA_COLLECTION_NAME,
    CHILD_METADATA_FIELDS,
    EMBEDDING_MODEL,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PORTFOLIO = ["INFY"]
CHILD_EXAMPLE_SPEAKER = "Salil Parekh"
CHILD_METADATA_SCHEMA = {field: "str" for field in CHILD_METADATA_FIELDS}
AGENT_CONNECTIONS = (
    "START → query_parser → research_analyst → financial_critic → END"
)


class AnalysisSwarm(Protocol):
    def stream_market_landscape(
        self,
        analysis_query: str,
        default_portfolio: list[str],
    ) -> Iterator[dict]: ...


def list_pdf_names(root: Path = ROOT_DIR) -> list[str]:
    """Return PDF filenames under financial_data, without changing the files."""
    data_dir = root / "financial_data"
    if not data_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_parent_lookup_summary(root: Path = ROOT_DIR) -> dict[str, int | bool]:
    """Read the lookup map only to validate its shape and count its entries."""
    lookup = load_parent_lookup(root)
    if lookup is None:
        return {"exists": False, "entry_count": 0}

    return {"exists": True, "entry_count": len(lookup)}


def load_parent_lookup(root: Path = ROOT_DIR) -> dict[str, str] | None:
    """Load and validate the read-only parent lookup map."""
    lookup_path = root / "parent_lookup.json"
    if not lookup_path.is_file():
        return None

    with lookup_path.open("r", encoding="utf-8") as lookup_file:
        lookup = json.load(lookup_file)
    if not isinstance(lookup, dict):
        raise ValueError("parent_lookup.json must contain a JSON object.")
    if any(not isinstance(parent_id, str) or not isinstance(text, str)
           for parent_id, text in lookup.items()):
        raise ValueError("parent_lookup.json must map string parent IDs to string text.")
    return lookup


def load_child_chunk_example(root: Path = ROOT_DIR) -> dict | None:
    """Read a child chunk belonging to the longest available speaker turn."""
    import chromadb

    client = chromadb.PersistentClient(path=str(root / "chroma_db"))
    collection_names = {collection.name for collection in client.list_collections()}
    if CHROMA_COLLECTION_NAME not in collection_names:
        return None

    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    parent_lookup = load_parent_lookup(root)
    if not parent_lookup:
        return None

    parent_records = collection.get(
        where={"speaker": CHILD_EXAMPLE_SPEAKER},
        include=["metadatas"],
    )
    metadatas = parent_records.get("metadatas") or []
    matching_parents = {
        metadata["parent_id"]
        for metadata in metadatas
        if metadata
        and metadata.get("speaker") == CHILD_EXAMPLE_SPEAKER
        and metadata.get("parent_id") in parent_lookup
    }
    if not matching_parents:
        return None

    parent_id = max(matching_parents, key=lambda key: len(parent_lookup[key]))
    records = collection.get(
        where={"parent_id": parent_id},
        limit=1,
        include=["documents", "metadatas"],
    )
    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    if not ids:
        return None
    if len(documents) != len(ids) or len(metadatas) != len(ids):
        raise ValueError("ChromaDB returned an incomplete child-chunk example.")

    metadata = metadatas[0]
    if not isinstance(metadata, dict):
        raise ValueError("ChromaDB returned invalid metadata for its child-chunk example.")
    if metadata.get("speaker") != CHILD_EXAMPLE_SPEAKER:
        raise ValueError(
            f"Selected child chunk is not attributed to {CHILD_EXAMPLE_SPEAKER}."
        )
    missing_fields = set(CHILD_METADATA_FIELDS) - metadata.keys()
    if missing_fields:
        raise ValueError(
            "ChromaDB child-chunk example is missing metadata fields: "
            + ", ".join(sorted(missing_fields))
        )
    return {
        "id": ids[0],
        "document": documents[0],
        "metadata": {
            field: metadata[field] for field in CHILD_METADATA_FIELDS
        },
    }


def parent_text_excerpt(text: str, max_length: int = 1350) -> str:
    """Keep the displayed real parent text example concise."""
    excerpt = text[:max_length].rstrip()
    return f"{excerpt}..."


def run_analysis_query(
    query: str,
    swarm_factory: Callable[[], AnalysisSwarm],
) -> Iterator[dict]:
    """Return a lazy stream of graph node updates and the final report."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Enter a query before running the analysis.")

    swarm = swarm_factory()
    return swarm.stream_market_landscape(
        normalized_query,
        DEFAULT_PORTFOLIO,
    )


def format_intermediate_outputs(node_updates: list[dict]) -> str:
    """Format each node's state delta without duplicating the final report."""
    formatted_nodes = []
    for update in node_updates:
        node_name = update["node"]
        state_update = update["state_update"]
        if state_update:
            output = json.dumps(state_update, indent=2, ensure_ascii=False, default=str)
        elif node_name == "financial_critic":
            output = "Final report is shown in the separate output box below."
        else:
            output = "No intermediate state updates."
        formatted_nodes.append(f"=== {node_name} ===\n{output}")
    return "\n\n".join(formatted_nodes)


@st.cache_resource
def get_swarm():
    from core_rag.agents_graph import LangGraphFinancialSwarm

    return LangGraphFinancialSwarm(
        lookup_path=str(ROOT_DIR / "parent_lookup.json"),
        db_dir=str(ROOT_DIR / "chroma_db"),
    )


def render_inventory() -> None:
    st.header("Project structures & data")
    documents_tab, child_tab, parent_tab, agents_tab, models_tab, queries_tab = st.tabs(
        ["PDF documents", "Child chunks", "Parent lookup", "Agent graph", "Models", "Supported queries"]
    )

    with documents_tab:
        st.subheader("Available PDF documents")
        pdf_names = list_pdf_names()
        if pdf_names:
            st.markdown("\n".join(f"- `{name}`" for name in pdf_names))
        else:
            st.info("No PDF documents found under `financial_data/`.")

    with child_tab:
        st.subheader("ChromaDB child-chunk record")
        st.caption(f"Collection: `{CHROMA_COLLECTION_NAME}`")
        st.code(
            "ChildChunk = {\n"
            "    id: str,                  # child UUID\n"
            "    document: str,            # enriched_text\n"
            "    metadata: {\n"
            + "".join(
                f"        {field}: {type_hint},\n"
                for field, type_hint in CHILD_METADATA_SCHEMA.items()
            )
            + "    },\n"
            "    embedding: list[float],   # generated locally\n"
            "}",
            language="python",
        )
        st.caption(
            f"Embedding model: `{EMBEDDING_MODEL}`. "
            "This is a schema view; individual vectors are not listed or modified."
        )
        try:
            child_example = load_child_chunk_example()
        except (OSError, ValueError) as error:
            st.error(f"Could not read a child-chunk example from ChromaDB: {error}")
            child_example = None
        if child_example:
            st.markdown(
                f"**Example — one real child chunk ({CHILD_EXAMPLE_SPEAKER})**"
            )
            st.json({
                **child_example,
                "embedding": "384-dimensional vector (omitted)",
            })
        else:
            st.info(
                f"No child-chunk example for {CHILD_EXAMPLE_SPEAKER} is available "
                "in the ChromaDB collection."
            )

    with parent_tab:
        st.subheader("Parent chunk lookup")
        st.code(
            "parent_lookup.json: dict[str, str]\n"
            "    key   = parent_id (UUID)\n"
            "    value = parent_text",
            language="python",
        )
        try:
            lookup = load_parent_lookup()
        except (OSError, json.JSONDecodeError, ValueError) as error:
            st.error(f"Could not inspect `parent_lookup.json`: {error}")
        else:
            if lookup is not None:
                st.caption(f"{len(lookup):,} parent text entries (read-only).")
                if child_example:
                    parent_id = child_example["metadata"].get("parent_id")
                    parent_text = lookup.get(parent_id)
                    if parent_id and parent_text:
                        st.markdown("**Example — matching real parent lookup entry**")
                        st.json({
                            parent_id: parent_text_excerpt(parent_text),
                        })
                    else:
                        st.info(
                            "The selected child chunk has no matching parent text "
                            "in `parent_lookup.json`."
                        )
                else:
                    st.info("No child chunk is available to match to a parent entry.")
            else:
                st.info("`parent_lookup.json` was not found.")

    with agents_tab:
        st.subheader("LangGraph agent connections")
        st.code(AGENT_CONNECTIONS, language="text")
        st.caption("Nodes: query_parser → research_analyst → financial_critic")

    with models_tab:
        st.subheader("Models in use")
        st.markdown(
            f"- **Query parser & research analyst:** `{LLM_MODEL}`\n"
            f"- **Financial critic / final report:** `{CRITIC_LLM_MODEL}`\n"
            f"- **Local embedding model:** `{EMBEDDING_MODEL}`"
        )

    with queries_tab:
        st.subheader("Supported queries")
        queries_path = ROOT_DIR / "supported_queries.md"
        try:
            supported_queries = queries_path.read_text(encoding="utf-8")
        except OSError as error:
            st.error(f"Could not read `supported_queries.md`: {error}")
        else:
            st.markdown(supported_queries)


def main() -> None:
    st.set_page_config(page_title="RAG-Based Concall Analyzer", layout="wide")
    st.title("RAG-Based Concall Analyzer")

    with st.form("analysis_query_form"):
        query = st.text_area(
            "Enter your analysis query",
            placeholder="For example: Analyze the impact of AI on the company's performance in 2025 and 2026.",
            height=110,
        )
        submitted = st.form_submit_button("Run analysis", type="primary")

    st.subheader("Intermediate agent outputs")
    intermediate_placeholder = st.empty()
    st.subheader("Final report")
    final_report_placeholder = st.empty()

    if submitted:
        if not query.strip():
            st.error("Enter a query before running the analysis.")
        else:
            st.session_state.pop("intermediate_outputs", None)
            st.session_state.pop("final_report", None)
            st.session_state.pop("last_query", None)
            node_updates = []
            st.session_state["last_query"] = query.strip()
            intermediate_placeholder.info("Waiting for the first agent node...")
            final_report_placeholder.info(
                "The final report will appear after the financial critic completes."
            )
            with st.spinner("Running the local agent graph and searching ChromaDB..."):
                for event in run_analysis_query(query, get_swarm):
                    if "final_report" in event:
                        st.session_state["final_report"] = event["final_report"]
                        final_report_placeholder.code(
                            event["final_report"],
                            language="markdown",
                            wrap_lines=True,
                        )
                        continue

                    node_updates.append(event)
                    st.session_state["intermediate_outputs"] = (
                        format_intermediate_outputs(node_updates)
                    )
                    intermediate_placeholder.code(
                        st.session_state["intermediate_outputs"],
                        language="text",
                        wrap_lines=True,
                    )
    else:
        if "intermediate_outputs" in st.session_state:
            st.caption(
                f"Query: {st.session_state.get('last_query', '')} — "
                "each section shows a completed node's state update."
            )
            intermediate_placeholder.code(
                st.session_state["intermediate_outputs"],
                language="text",
                wrap_lines=True,
            )
        else:
            intermediate_placeholder.info(
                "Submit a query to display the intermediate agent outputs."
            )

        if "final_report" in st.session_state:
            final_report_placeholder.code(
                st.session_state["final_report"],
                language="markdown",
                wrap_lines=True,
            )
        else:
            final_report_placeholder.info(
                "Submit a query to display the final report from the financial critic agent."
            )

    st.divider()
    render_inventory()


if __name__ == "__main__":
    main()
