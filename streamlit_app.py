import json
from pathlib import Path
from typing import Callable, Protocol

import streamlit as st

from core_rag.agents_graph import CRITIC_LLM_MODEL, LLM_MODEL
from core_rag.vector_store import (
    CHROMA_COLLECTION_NAME,
    CHILD_METADATA_FIELDS,
    EMBEDDING_MODEL,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PORTFOLIO = ["INFY"]
CHILD_METADATA_SCHEMA = {field: "str" for field in CHILD_METADATA_FIELDS}
AGENT_CONNECTIONS = (
    "START → query_parser → research_analyst → financial_critic → END"
)


class AnalysisSwarm(Protocol):
    def analyze_market_landscape(
        self,
        analysis_query: str,
        default_portfolio: list[str],
    ) -> str: ...


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
    lookup_path = root / "parent_lookup.json"
    if not lookup_path.is_file():
        return {"exists": False, "entry_count": 0}

    with lookup_path.open("r", encoding="utf-8") as lookup_file:
        lookup = json.load(lookup_file)
    if not isinstance(lookup, dict):
        raise ValueError("parent_lookup.json must contain a JSON object.")
    if any(not isinstance(parent_id, str) or not isinstance(text, str)
           for parent_id, text in lookup.items()):
        raise ValueError("parent_lookup.json must map string parent IDs to string text.")
    return {"exists": True, "entry_count": len(lookup)}


def run_analysis_query(
    query: str,
    swarm_factory: Callable[[], AnalysisSwarm],
) -> str:
    """Run the existing graph against the fixed portfolio, without ingesting data."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Enter a query before running the analysis.")

    swarm = swarm_factory()
    return swarm.analyze_market_landscape(normalized_query, DEFAULT_PORTFOLIO)


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

    with parent_tab:
        st.subheader("Parent chunk lookup")
        st.code(
            "parent_lookup.json: dict[str, str]\n"
            "    key   = parent_id (UUID)\n"
            "    value = parent_text",
            language="python",
        )
        try:
            summary = load_parent_lookup_summary()
        except (OSError, json.JSONDecodeError, ValueError) as error:
            st.error(f"Could not inspect `parent_lookup.json`: {error}")
        else:
            if summary["exists"]:
                st.caption(f"{summary['entry_count']:,} parent text entries (read-only).")
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

    if submitted:
        if not query.strip():
            st.error("Enter a query before running the analysis.")
        else:
            with st.spinner("Running the local agent graph and searching ChromaDB..."):
                report = run_analysis_query(query, get_swarm)
            st.session_state["final_report"] = report
            st.session_state["last_query"] = query.strip()

    st.subheader("Final report")
    if "final_report" in st.session_state:
        st.caption(f"Query: {st.session_state['last_query']}")
        st.code(st.session_state["final_report"], language="markdown", wrap_lines=True)
    else:
        st.info("Submit a query to display the final report from the financial critic agent.")

    st.divider()
    render_inventory()


if __name__ == "__main__":
    main()
