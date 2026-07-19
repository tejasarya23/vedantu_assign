from __future__ import annotations

from datetime import date

import pytest

from study_assistant import StudyAssistant
from study_assistant.models import AssistantResponse, SearchDocument, SearchResult, ToolResult


class FakeStudyAgent:
    def run(self, query: str, *, today: date) -> AssistantResponse:
        weak_topics = ["Algebra", "Quadratic Equations", "Light - Reflection and Refraction"]
        if "roots" in query.lower():
            weak_topics = ["Quadratic Equations", "Algebra", "Light - Reflection and Refraction"]

        tool_calls = [
            ToolResult(
                tool_name="semantic_search_student_data",
                payload=[{"title": "Weak topic: Quadratic Equations", "score": 0.42}],
                reasoning="LangGraph ReAct called semantic retrieval first.",
            ),
            ToolResult(
                tool_name="get_weak_topics",
                payload=weak_topics,
                reasoning="Retrieved profile weak topics.",
            ),
            ToolResult(
                tool_name="get_upcoming_tests",
                payload=[
                    {
                        "test_name": "Math Weekly Test",
                        "date": "2026-04-14",
                        "topics": ["Algebra", "Quadratic Equations"],
                    }
                ],
                reasoning="Retrieved upcoming tests.",
            ),
            ToolResult(
                tool_name="recommend_study_material",
                payload=[
                    {
                        "material_id": "M103",
                        "topic": "Quadratic Equations",
                        "title": "Quadratic Equations Concept Video",
                    }
                ],
                reasoning="Recommended semantically matched materials.",
            ),
        ]
        context = [
            SearchResult(
                document=SearchDocument(
                    doc_id="weak-topic:Quadratic Equations",
                    kind="weak_topic",
                    title="Weak topic: Quadratic Equations",
                    text="Arjun is weak in Quadratic Equations.",
                    metadata={"topic": "Quadratic Equations"},
                ),
                score=0.42,
            )
        ]
        return AssistantResponse(
            answer=(
                "Arjun should prepare for Math Weekly Test by revising Algebra and "
                "Quadratic Equations. Semantic retrieval matched equation roots to "
                "Quadratic Equations."
            ),
            tool_calls=tool_calls,
            retrieved_context=context,
        )


def test_semantic_query_for_maths_test_finds_mathematics_test() -> None:
    assistant = StudyAssistant(agent=FakeStudyAgent())

    response = assistant.answer(
        "I have a Maths exam coming up. Help me prepare.",
        today=date(2026, 4, 10),
    )

    test_payload = next(
        result.payload
        for result in response.tool_calls
        if result.tool_name == "get_upcoming_tests"
    )
    assert test_payload[0]["test_name"] == "Math Weekly Test"
    assert "Algebra" in response.answer
    assert "Quadratic Equations" in response.answer


def test_weak_area_query_returns_personalized_topic_and_material() -> None:
    assistant = StudyAssistant(agent=FakeStudyAgent())

    response = assistant.answer(
        "I struggle with equation roots. What should I do next?",
        today=date(2026, 4, 10),
    )

    material_payload = next(
        result.payload
        for result in response.tool_calls
        if result.tool_name == "recommend_study_material"
    )
    assert "Arjun" in response.answer
    assert any(item["topic"] == "Quadratic Equations" for item in material_payload)
    assert "Semantic retrieval matched" in response.answer


def test_response_contains_langgraph_tool_reasoning() -> None:
    assistant = StudyAssistant(agent=FakeStudyAgent())

    response = assistant.answer("What are my weak areas?", today=date(2026, 4, 10))

    tool_names = {result.tool_name for result in response.tool_calls}
    assert "semantic_search_student_data" in tool_names
    assert "get_weak_topics" in tool_names
    assert "recommend_study_material" in tool_names
    assert all(result.reasoning for result in response.tool_calls)


def test_missing_openai_key_raises_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assistant = StudyAssistant()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        assistant.answer("What should I study this week?", today=date(2026, 4, 10))
