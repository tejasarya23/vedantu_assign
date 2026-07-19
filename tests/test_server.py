from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from study_assistant.models import AssistantResponse, SearchDocument, SearchResult, ToolResult
from study_assistant.server import create_app


class FakeStudyAgent:
    def run(self, query: str, *, today: date) -> AssistantResponse:
        return AssistantResponse(
            answer=(
                "Arjun should prepare for Math Weekly Test by revising Algebra and "
                "Quadratic Equations. AI generated this after LangGraph ReAct tool calls."
            ),
            tool_calls=[
                ToolResult(
                    tool_name="semantic_search_student_data",
                    payload=[{"title": "Math Weekly Test", "score": 0.65}],
                    reasoning="LangGraph ReAct called semantic search first.",
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
                    reasoning="LangGraph ReAct retrieved upcoming tests.",
                ),
            ],
            retrieved_context=[
                SearchResult(
                    document=SearchDocument(
                        doc_id="test:T201",
                        kind="test",
                        title="Math Weekly Test",
                        text="Math Weekly Test covers Algebra and Quadratic Equations.",
                        metadata={"test_id": "T201"},
                    ),
                    score=0.65,
                )
            ],
        )


def test_fastapi_ask_endpoint_returns_grounded_response() -> None:
    client = TestClient(create_app(agent=FakeStudyAgent()))

    response = client.post(
        "/ask",
        json={
            "query": "I have a Maths test coming up. Help me prepare.",
            "today": "2026-04-10",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Math Weekly Test" in payload["answer"]
    assert "LangGraph ReAct" in payload["answer"]
    assert any(tool["tool_name"] == "get_upcoming_tests" for tool in payload["tool_calls"])


def test_fastapi_health_reports_openai_status() -> None:
    client = TestClient(create_app(agent=FakeStudyAgent()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_fastapi_ask_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={
            "query": "What should I study this week?",
            "today": "2026-04-10",
        },
    )

    assert response.status_code == 503
    assert "OPENAI_API_KEY is required" in response.json()["detail"]
