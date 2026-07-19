from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class StudentProfile:
    student_id: str
    name: str
    grade: int
    board: str
    target_exam: str
    daily_study_time_minutes: int
    strong_topics: list[str]
    weak_topics: list[str]


@dataclass(frozen=True)
class SubjectPerformance:
    subject: str
    overall_score_percentage: int


@dataclass(frozen=True)
class StudyMaterial:
    material_id: str
    topic: str
    title: str


@dataclass(frozen=True)
class UpcomingTest:
    test_id: str
    subject: str
    test_name: str
    date: date
    topics: list[str]


@dataclass(frozen=True)
class StudentData:
    profile: StudentProfile
    performance: list[SubjectPerformance]
    materials: list[StudyMaterial]
    tests: list[UpcomingTest]


@dataclass(frozen=True)
class SearchDocument:
    doc_id: str
    kind: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    document: SearchDocument
    score: float


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    payload: Any
    reasoning: str


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    tool_calls: list[ToolResult]
    retrieved_context: list[SearchResult]

