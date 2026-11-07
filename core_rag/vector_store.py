import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv

CHROMA_COLLECTION_NAME = "infosys_financial_transcripts"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHILD_METADATA_FIELDS = (
    "parent_id",
    "ticker",
    "year",
    "month",
    "index",
    "speaker",
    "speaker_number",
    "page_number",
)


class LocalVectorStorage:
    def __init__(self, db_dir: str = "chroma_db"):
        self.db_dir = db_dir
        load_dotenv()  # Load HF_TOKEN
        
        # 1. Initialize local persistent Chroma client (Saves files directly to your hard drive)
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # 2. Setup the free BGE Embedding Function (Automatically downloads and runs model locally)
        print("🧠 Loading local BAAI/bge-small-en-v1.5 embedding model...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # 3. Create or fetch our dynamic collection array
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        print("✅ Local Vector Database & Embedding Layer Connected.")

    def index_child_chunks(self, child_chunks: list[dict]):
        """
        Takes our enriched child chunks and upserts them into ChromaDB 
        along with their metadata filtering payloads.
        """
        if not child_chunks:
            return

        ids = []
        documents = []
        metadatas = []

        print(f"📦 Preparing {len(child_chunks)} child vectors for indexing...")
        for child in child_chunks:
            ids.append(child["child_id"])
            # The database embeds the text found inside this field natively
            documents.append(child["enriched_text"])
            
            # Pack all attributes we want to filter on later into the metadata dict
            metadatas.append({
                field: str(child[field])
                for field in CHILD_METADATA_FIELDS
            })

        # Batch upload vectors to the local database cluster
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"💾 Vector sync complete! Chunks successfully committed to './{self.db_dir}' folder.")

    def query_database(self, query_text: str, ticker: str = None, year: str = None, top_k: int = 3) -> dict:
        """
        Queries the database using semantic search while applying optional hard metadata filters.
        """
        # Construct hard filtering constraints if provided
        where_filter = {}
        if ticker: where_filter["ticker"] = ticker
        if year: where_filter["year"] = str(year)

        # Run vector search query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        return results

    def get_parent_by_speaker_number(
        self,
        ticker: str,
        year: str,
        month: str,
        index: str,
        speaker_number: int
    ) -> dict | None:
        """Return metadata for the unique parent at a position in one transcript."""
        where_filter = {
            "$and": [
                {"ticker": str(ticker)},
                {"year": str(year)},
                {"month": str(month)},
                {"index": str(index)},
                {"speaker_number": str(speaker_number)}
            ]
        }
        results = self.collection.get(
            where=where_filter,
            include=["metadatas"]
        )

        metadatas = results.get("metadatas") or []
        if not metadatas:
            return None

        parent_ids = {metadata.get("parent_id") for metadata in metadatas if metadata}
        if len(parent_ids) != 1 or None in parent_ids:
            raise ValueError(
                "Expected one parent for transcript speaker number "
                f"{speaker_number}, found {len(parent_ids)}."
            )
        return next(metadata for metadata in metadatas if metadata)

    def get_company_timeline(self, ticker: str) -> list[dict[str, str]]:
        """
        Returns the company's available transcript months, newest first.
        """
        results = self.collection.get(
            where={"ticker": ticker},
            include=["metadatas"]
        )

        months = set()
        for metadata in results.get("metadatas") or []:
            if not metadata:
                continue

            year = str(metadata.get("year", ""))
            month = str(metadata.get("month", ""))
            if len(year) == 4 and year.isdigit() and month.isdigit():
                month_number = int(month)
                if 1 <= month_number <= 12:
                    months.add((int(year), month_number))

        return [
            {"key": f"{year:04d}_{month:02d}"}
            for year, month in sorted(months, reverse=True)
        ]

    def get_company_extreme_document(
        self,
        ticker: str,
        latest: bool
    ) -> dict[str, str] | None:
        """
        Returns the latest or earliest available transcript descriptor for a ticker.

        For a transcript month containing multiple indexed calls, the lowest numeric
        index is selected.
        """
        results = self.collection.get(
            where={"ticker": ticker},
            include=["metadatas"]
        )

        documents = []
        seen_documents = set()
        for metadata in results.get("metadatas") or []:
            if not metadata:
                continue

            year = str(metadata.get("year", ""))
            month = str(metadata.get("month", ""))
            index = str(metadata.get("index", ""))
            if (
                len(year) != 4
                or not (year.isascii() and year.isdecimal())
                or len(month) != 2
                or not (month.isascii() and month.isdecimal())
                or not (index.isascii() and index.isdecimal())
                or int(year) < 1
                or not 1 <= int(month) <= 12
            ):
                continue

            document = (int(year), int(month), int(index), year, f"{int(month):02d}", index)
            identity = document[:3]
            if identity not in seen_documents:
                seen_documents.add(identity)
                documents.append(document)

        if not documents:
            return None

        selected_month = (max if latest else min)(
            (document[0], document[1]) for document in documents
        )
        selected_document = min(
            (
                document for document in documents
                if (document[0], document[1]) == selected_month
            ),
            key=lambda document: (document[2], document[5])
        )
        return {
            "year": selected_document[3],
            "month": selected_document[4],
            "index": selected_document[5]
        }

    def query_database_advanced(
        self,
        query_text: str,
        ticker: str,
        target_keys: list[str],
        top_k: int,
        target_index: str | None = None
    ) -> dict:
        """
        Performs semantic search for a ticker, optionally limited to YYYY_MM keys
        and a transcript index.

        Results are flattened from Chroma's single-query batch shape for callers
        that consume each matched metadata record directly.
        """
        filters = [{"ticker": ticker}]
        if target_keys:
            year_months = []
            for key in dict.fromkeys(target_keys):
                parts = key.split("_")
                if (
                    len(parts) != 2
                    or len(parts[0]) != 4
                    or not parts[0].isdigit()
                    or not parts[1].isdigit()
                    or not 1 <= int(parts[1]) <= 12
                ):
                    raise ValueError(f"Invalid timeline key {key!r}; expected YYYY_MM.")
                year_months.append((parts[0], f"{int(parts[1]):02d}"))

            month_filters = [
                {"$and": [{"year": year}, {"month": month}]}
                for year, month in year_months
            ]
            selected_months = (
                month_filters[0]
                if len(month_filters) == 1
                else {"$or": month_filters}
            )
            filters.append(selected_months)
        if target_index is not None:
            target_index = str(target_index)
            if not (target_index.isascii() and target_index.isdecimal()):
                raise ValueError(f"Invalid target index {target_index!r}; expected a numeric index.")
            filters.append({"index": target_index})

        where_filter = filters[0] if len(filters) == 1 else {"$and": filters}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )
        for field in ("ids", "documents", "metadatas", "distances", "embeddings"):
            values = results.get(field)
            if values is not None and len(values) == 1:
                results[field] = values[0]
        return results
