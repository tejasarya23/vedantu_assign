from __future__ import annotations

from datetime import date
from pathlib import Path

from study_assistant.data_loader import DEFAULT_DATA_DIR, load_student_data
from study_assistant.langgraph_agent import LangGraphStudyAgent, StudyAgentRunner
from study_assistant.models import AssistantResponse, StudentData
from study_assistant.semantic_search import SemanticSearchIndex, build_documents
from study_assistant.tools import StudentTools


class StudyAssistant:
    def __init__(
        self,
        data: StudentData | None = None,
        *,
        agent: StudyAgentRunner | None = None,
    ) -> None:
        self.data = data or load_student_data(DEFAULT_DATA_DIR)
        self.search_index = SemanticSearchIndex(build_documents(self.data))
        self.tools = StudentTools(self.data, self.search_index)
        self.agent = agent or LangGraphStudyAgent(
            data=self.data,
            search_index=self.search_index,
            tools=self.tools,
        )

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        *,
        agent: StudyAgentRunner | None = None,
    ) -> "StudyAssistant":
        return cls(load_student_data(Path(data_dir)), agent=agent)

    def answer(
        self,
        query: str,
        *,
        today: date | None = None,
    ) -> AssistantResponse:
        today = today or date.today()
        return self.agent.run(query, today=today)
