"""Knowledge Base Construction module."""

from __future__ import annotations

import json
from pathlib import Path

import faiss


class StructuredComponentKnowledgeBase:
    """Structured component database with FAISS retrieval index."""

    def __init__(self, components_json: Path, embedding_model) -> None:
        with components_json.open("r", encoding="utf-8") as handle:
            self._database = json.load(handle)

        self._records = [json.dumps(item) for item in self._database.values()]
        self._embedding_model = embedding_model

        embeddings = embedding_model.encode(self._records)
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve_by_label(self, label: str, top_k: int) -> list[str]:
        """Retrieve database entries semantically close to one object label."""
        label_embedding = self._embedding_model.encode([label])
        _, indices = self._index.search(label_embedding, k=top_k)
        return [self._records[idx] for idx in indices[0]]

