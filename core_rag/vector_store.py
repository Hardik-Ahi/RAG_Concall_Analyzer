import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv

class LocalVectorStorage:
    def __init__(self, db_dir: str = "chroma_db"):
        self.db_dir = db_dir
        load_dotenv()  # Load HF_TOKEN
        
        # 1. Initialize local persistent Chroma client (Saves files directly to your hard drive)
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # 2. Setup the free BGE Embedding Function (Automatically downloads and runs model locally)
        print("🧠 Loading local BAAI/bge-small-en-v1.5 embedding model...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # 3. Create or fetch our dynamic collection array
        self.collection = self.client.get_or_create_collection(
            name="infosys_financial_transcripts",
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
                "parent_id": child["parent_id"],
                "ticker": child["ticker"],
                "year": str(child["year"]),
                "month": str(child["month"]),
                "index": str(child["index"]),
                "speaker": child["speaker"]
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

    def query_database_advanced(
        self,
        query_text: str,
        ticker: str,
        target_keys: list[str],
        top_k: int
    ) -> dict:
        """
        Performs semantic search for a ticker, optionally limited to YYYY_MM keys.

        Results are flattened from Chroma's single-query batch shape for callers
        that consume each matched metadata record directly.
        """
        where_filter = {"ticker": ticker}
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
            where_filter = {"$and": [{"ticker": ticker}, selected_months]}

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
