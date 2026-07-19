from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from study_assistant.models import AssistantResponse, SearchResult, StudentData, ToolResult
from study_assistant.semantic_search import SemanticSearchIndex
from study_assistant.tools import StudentTools


load_dotenv()


class StudyAgentRunner(Protocol):
    def run(self, query: str, *, today: date) -> AssistantResponse:
        ...


class LangGraphStudyAgent:
    def __init__(
        self,
        *,
        data: StudentData,
        search_index: SemanticSearchIndex,
        tools: StudentTools,
        model: str | None = None,
    ) -> None:
        self.data = data
        self.search_index = search_index
        self.tools = tools
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.6")

    def run(self, query: str, *, today: date) -> AssistantResponse:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required because LangGraph ReAct generation is mandatory.")

        tool_results: list[ToolResult] = []
        retrieved_context = self.search_index.search(query, top_k=6)
        langgraph_tools = self._build_langgraph_tools(
            today=today,
            retrieved_context=retrieved_context,
            tool_results=tool_results,
        )
        model = ChatOpenAI(model=self.model)
        agent = create_react_agent(
            model,
            tools=langgraph_tools,
            prompt=self._system_prompt(today),
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )
        answer = self._final_message_content(result)
        if not answer:
            raise RuntimeError("LangGraph ReAct agent returned an empty response.")

        return AssistantResponse(
            answer=answer,
            tool_calls=tool_results,
            retrieved_context=retrieved_context,
        )

    def _build_langgraph_tools(
        self,
        *,
        today: date,
        retrieved_context: list[SearchResult],
        tool_results: list[ToolResult],
    ) -> list[Any]:
        student_tools = self.tools
        search_index = self.search_index

        @tool
        def semantic_search_student_data(query: str, top_k: int = 5) -> str:
            """Find semantically relevant student records when wording does not exactly match dataset terms."""
            capped_top_k = max(1, min(top_k, 8))
            results = search_index.search(query, top_k=capped_top_k)
            payload = [
                {
                    "doc_id": result.document.doc_id,
                    "kind": result.document.kind,
                    "title": result.document.title,
                    "text": result.document.text,
                    "score": round(result.score, 4),
                    "metadata": result.document.metadata,
                }
                for result in results
            ]
            tool_results.append(
                ToolResult(
                    tool_name="semantic_search_student_data",
                    payload=payload,
                    reasoning=(
                        "LangGraph ReAct called semantic retrieval to map the query to related "
                        "student records, topics, materials, tests, or performance evidence."
                    ),
                )
            )
            if not retrieved_context:
                retrieved_context.extend(results)
            return json.dumps(payload, indent=2)

        @tool
        def get_weak_topics(query: str | None = None) -> str:
            """Return the student's weak topics, semantically ranked when a query is provided."""
            result = student_tools.get_weak_topics(query)
            tool_results.append(result)
            return json.dumps({"topics": result.payload, "reasoning": result.reasoning}, indent=2)

        @tool
        def get_upcoming_tests(query: str | None = None) -> str:
            """Return upcoming or semantically matched tests using the request date as today's date."""
            result = student_tools.get_upcoming_tests(query, today=today)
            tool_results.append(result)
            return json.dumps({"tests": result.payload, "reasoning": result.reasoning}, indent=2)

        @tool
        def get_subject_performance() -> str:
            """Return exact subject-level score history."""
            result = student_tools.get_subject_performance()
            tool_results.append(result)
            return json.dumps({"performance": result.payload, "reasoning": result.reasoning}, indent=2)

        @tool
        def recommend_study_material(query: str, topics: list[str] | None = None, limit: int = 3) -> str:
            """Recommend study materials using semantic search and optional topic filters."""
            result = student_tools.recommend_study_material(query, topics=topics, limit=limit)
            tool_results.append(result)
            return json.dumps({"materials": result.payload, "reasoning": result.reasoning}, indent=2)

        return [
            semantic_search_student_data,
            get_weak_topics,
            get_upcoming_tests,
            get_subject_performance,
            recommend_study_material,
        ]

    def _system_prompt(self, today: date) -> str:
        profile = self.data.profile
        return f"""
You are Arjun's study assistant.

Student profile:
- student_id: {profile.student_id}
- name: {profile.name}
- grade: {profile.grade}
- board: {profile.board}
- target_exam: {profile.target_exam}
- daily study time: {profile.daily_study_time_minutes} minutes

Today is {today.isoformat()}.

Use ReAct tool calling. You must inspect student data through tools before giving the final answer.
Call semantic_search_student_data first for every query. Then call the relevant structured tools:
- get_weak_topics for weak areas, priority, or improvement questions.
- get_upcoming_tests for weekly plans, test prep, or exam-related questions.
- get_subject_performance when ranking subjects or explaining performance.
- recommend_study_material before recommending resources.

Final answer requirements:
- Be personalized and specific to the student.
- Give a clear next action or study plan.
- Explain why each recommendation follows from tool results.
- Mention tool evidence naturally; do not invent scores, tests, topics, or materials.
- If a test date is before today's date, say it is historical evidence, not an upcoming deadline.
""".strip()

    def _final_message_content(self, result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        if not messages:
            return ""
        final_message = messages[-1]
        if isinstance(final_message, dict):
            content = final_message.get("content", "")
        else:
            content = getattr(final_message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            ]
            return "\n".join(parts).strip()
        return str(content).strip()
