from __future__ import annotations

import json
import importlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RetrievedSnippet:
    text: str
    route: str
    source: str


class RAGStore:
    def __init__(self, project_root: str | Path, persist_dir: str = ".chroma_party") -> None:
        self.project_root = Path(project_root)
        self.persist_path = self.project_root / persist_dir
        self._vectorstore = None

    def build(self, reset: bool = False) -> None:
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        warnings.filterwarnings(
            "ignore",
            message=r".*HuggingFaceEmbeddings.*deprecated.*",
        )

        if reset and self.persist_path.exists():
            for item in self.persist_path.glob("**/*"):
                if item.is_file():
                    item.unlink()

        documents = self._prepare_documents()
        if not documents:
            raise ValueError("No documents available to build RAG index")

        RecursiveCharacterTextSplitter = getattr(
            importlib.import_module("langchain.text_splitter"),
            "RecursiveCharacterTextSplitter",
        )
        HuggingFaceEmbeddings = getattr(
            importlib.import_module("langchain_community.embeddings"),
            "HuggingFaceEmbeddings",
        )
        Chroma = getattr(
            importlib.import_module("langchain_community.vectorstores"),
            "Chroma",
        )
        Settings = getattr(
            importlib.import_module("chromadb.config"),
            "Settings",
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=str(self.persist_path),
            collection_name="party_case",
            client_settings=Settings(anonymized_telemetry=False),
        )

    def retrieve(self, query: str, route: str = "evidence", k: int = 3) -> List[RetrievedSnippet]:
        if self._vectorstore is None:
            self.build(reset=False)

        filters = {"route": route} if route in {"alibi", "evidence", "noise"} else None

        results = self._vectorstore.similarity_search(query, k=k, filter=filters)
        snippets: List[RetrievedSnippet] = []
        for doc in results:
            snippets.append(
                RetrievedSnippet(
                    text=doc.page_content,
                    route=doc.metadata.get("route", "unknown"),
                    source=doc.metadata.get("source", "unknown"),
                )
            )
        return snippets

    def _prepare_documents(self):
        Document = getattr(importlib.import_module("langchain_core.documents"), "Document")

        story_json_path = self.project_root / "src" / "data" / "story" / "party_case.json"
        story_notes_path = self.project_root / "src" / "data" / "story" / "party_case_notes.md"
        noise_path = self.project_root / "src" / "data" / "noise" / "noise_master.md"

        payload = json.loads(story_json_path.read_text(encoding="utf-8"))

        docs = []
        for slot in payload.get("slots", []):
            docs.append(
                Document(
                    page_content=(
                        f"Time: {slot['time']} | Location: {slot['location']} | "
                        f"Action: {slot['action']} | Witnesses: {', '.join(slot.get('witnesses', []))}"
                    ),
                    metadata={"route": "alibi", "source": "party_case.json"},
                )
            )

        for event in payload.get("camera", []):
            docs.append(
                Document(
                    page_content=(
                        f"Camera Time: {event['time']} | Location: {event['location']} | "
                        f"Observed: {event['observed_action']}"
                    ),
                    metadata={"route": "evidence", "source": "party_case.json"},
                )
            )

        docs.append(
            Document(
                page_content=story_notes_path.read_text(encoding="utf-8"),
                metadata={"route": "evidence", "source": "party_case_notes.md"},
            )
        )
        docs.append(
            Document(
                page_content=noise_path.read_text(encoding="utf-8"),
                metadata={"route": "noise", "source": "noise_master.md"},
            )
        )

        return docs
