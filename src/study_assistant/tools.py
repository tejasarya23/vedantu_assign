from __future__ import annotations

from datetime import date

from study_assistant.models import StudyMaterial, StudentData, ToolResult, UpcomingTest
from study_assistant.semantic_search import SemanticSearchIndex


class StudentTools:
    def __init__(self, data: StudentData, search_index: SemanticSearchIndex) -> None:
        self.data = data
        self.search_index = search_index

    def get_weak_topics(self, query: str | None = None) -> ToolResult:
        weak_topics = list(self.data.profile.weak_topics)
        if query:
            semantic_matches = self.search_index.search(
                query,
                top_k=len(weak_topics),
                kind="weak_topic",
                min_score=0.03,
            )
            matched_topics = [
                result.document.metadata["topic"]
                for result in semantic_matches
                if result.document.metadata["topic"] in weak_topics
            ]
            weak_topics = matched_topics or weak_topics

        return ToolResult(
            tool_name="get_weak_topics",
            payload=weak_topics,
            reasoning=(
                "Used profile weak_topics, then semantic ranking when the query mentioned "
                "a concept or used wording like struggle/improve instead of an exact topic name."
            ),
        )

    def get_subject_performance(self) -> ToolResult:
        payload = [
            {
                "subject": item.subject,
                "overall_score_percentage": item.overall_score_percentage,
            }
            for item in self.data.performance
        ]
        return ToolResult(
            tool_name="get_subject_performance",
            payload=payload,
            reasoning=(
                "Used structured performance data because marks are exact numeric facts and "
                "should not be inferred from semantic similarity."
            ),
        )

    def get_upcoming_tests(
        self,
        query: str | None = None,
        *,
        today: date | None = None,
        include_past_when_no_future_match: bool = True,
    ) -> ToolResult:
        today = today or date.today()
        tests = sorted(self.data.tests, key=lambda item: item.date)
        future_tests = [test for test in tests if test.date >= today]

        if query:
            matched_results = self.search_index.search(query, top_k=5, kind="test", min_score=0.05)
            matched_ids = {result.document.metadata["test_id"] for result in matched_results}
            filtered = [test for test in future_tests if test.test_id in matched_ids]
            if not filtered and include_past_when_no_future_match:
                filtered = [test for test in tests if test.test_id in matched_ids]
            tests = filtered
        else:
            tests = future_tests or (tests if include_past_when_no_future_match else [])

        payload = [self._test_payload(test, today) for test in tests]
        return ToolResult(
            tool_name="get_upcoming_tests",
            payload=payload,
            reasoning=(
                "Used date filtering for upcoming work and semantic search for subject/test wording "
                "such as Maths, exam, assessment, or weekly test."
            ),
        )

    def recommend_study_material(
        self,
        query: str,
        *,
        topics: list[str] | None = None,
        limit: int = 3,
    ) -> ToolResult:
        topic_set = {topic.lower() for topic in topics or []}
        semantic_matches = self.search_index.search(query, top_k=limit * 3, kind="material")
        materials: list[StudyMaterial] = []

        for result in semantic_matches:
            topic = str(result.document.metadata["topic"])
            if not topic_set or topic.lower() in topic_set:
                materials.append(
                    StudyMaterial(
                        material_id=result.document.metadata["material_id"],
                        topic=topic,
                        title=result.document.metadata["title"],
                    )
                )
            if len(materials) >= limit:
                break

        if topics:
            by_topic = {material.topic.lower(): material for material in self.data.materials}
            for topic in topics:
                material = by_topic.get(topic.lower())
                if material and all(existing.material_id != material.material_id for existing in materials):
                    materials.append(material)
                if len(materials) >= limit:
                    break

        payload = [
            {
                "material_id": material.material_id,
                "topic": material.topic,
                "title": material.title,
            }
            for material in materials[:limit]
        ]
        return ToolResult(
            tool_name="recommend_study_material",
            payload=payload,
            reasoning=(
                "Used semantic search over material titles and topic descriptions so a query like "
                "'equation roots video' can still find Quadratic Equations material."
            ),
        )

    def _test_payload(self, test: UpcomingTest, today: date) -> dict[str, object]:
        days_until = (test.date - today).days
        if days_until > 0:
            timing = f"in {days_until} day(s)"
        elif days_until == 0:
            timing = "today"
        else:
            timing = f"{abs(days_until)} day(s) ago"

        return {
            "test_id": test.test_id,
            "subject": test.subject,
            "test_name": test.test_name,
            "date": test.date.isoformat(),
            "days_until": days_until,
            "timing": timing,
            "topics": test.topics,
        }

