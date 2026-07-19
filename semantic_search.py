from __future__ import annotations

from study_assistant.embeddings import LocalSemanticEmbedder, cosine_similarity
from study_assistant.models import SearchDocument, SearchResult, StudentData


def build_documents(data: StudentData) -> list[SearchDocument]:
    documents: list[SearchDocument] = []

    for topic in data.profile.weak_topics:
        documents.append(
            SearchDocument(
                doc_id=f"weak-topic:{topic}",
                kind="weak_topic",
                title=f"Weak topic: {topic}",
                text=(
                    f"{data.profile.name} is weak in {topic}. "
                    f"This is a priority topic for improvement and practice."
                ),
                metadata={"topic": topic},
            )
        )

    for topic in data.profile.strong_topics:
        documents.append(
            SearchDocument(
                doc_id=f"strong-topic:{topic}",
                kind="strong_topic",
                title=f"Strong topic: {topic}",
                text=(
                    f"{data.profile.name} is strong in {topic}. "
                    f"Use this as confidence, not the main remediation priority."
                ),
                metadata={"topic": topic},
            )
        )

    for performance in data.performance:
        documents.append(
            SearchDocument(
                doc_id=f"performance:{performance.subject}",
                kind="performance",
                title=f"{performance.subject} performance",
                text=(
                    f"{data.profile.name}'s {performance.subject} score is "
                    f"{performance.overall_score_percentage} percent. Low scores need "
                    f"more revision time and focused practice."
                ),
                metadata={
                    "subject": performance.subject,
                    "score": performance.overall_score_percentage,
                },
            )
        )

    for material in data.materials:
        documents.append(
            SearchDocument(
                doc_id=f"material:{material.material_id}",
                kind="material",
                title=material.title,
                text=(
                    f"Study material {material.title} covers {material.topic}. "
                    f"It can be used for revision, concept clarity, and practice."
                ),
                metadata={
                    "material_id": material.material_id,
                    "topic": material.topic,
                    "title": material.title,
                },
            )
        )

    for test in data.tests:
        documents.append(
            SearchDocument(
                doc_id=f"test:{test.test_id}",
                kind="test",
                title=test.test_name,
                text=(
                    f"{test.test_name} is a {test.subject} test on {test.date.isoformat()} "
                    f"covering {', '.join(test.topics)}. Prepare by revising tested topics."
                ),
                metadata={
                    "test_id": test.test_id,
                    "subject": test.subject,
                    "date": test.date.isoformat(),
                    "topics": test.topics,
                },
            )
        )

    return documents


class SemanticSearchIndex:
    def __init__(
        self,
        documents: list[SearchDocument],
        embedder: LocalSemanticEmbedder | None = None,
    ) -> None:
        self.documents = documents
        self.embedder = embedder or LocalSemanticEmbedder()
        self._embeddings = [
            self.embedder.embed(f"{document.title}. {document.text}")
            for document in self.documents
        ]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        kind: str | None = None,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        query_embedding = self.embedder.embed(query)
        results = [
            SearchResult(document=document, score=cosine_similarity(query_embedding, embedding))
            for document, embedding in zip(self.documents, self._embeddings)
            if kind is None or document.kind == kind
        ]
        return [
            result
            for result in sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
            if result.score >= min_score
        ]

