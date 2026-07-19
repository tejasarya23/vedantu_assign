from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from study_assistant.models import (
    StudentData,
    StudentProfile,
    StudyMaterial,
    SubjectPerformance,
    UpcomingTest,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_student_data(data_dir: Path | str = DEFAULT_DATA_DIR) -> StudentData:
    data_path = Path(data_dir)
    profile_data = _read_json(data_path / "student_profile.json")
    performance_data = _read_json(data_path / "performance_history.json")
    materials_data = _read_json(data_path / "study_materials.json")
    tests_data = _read_json(data_path / "upcoming_tests.json")

    profile = StudentProfile(**profile_data)
    performance = [
        SubjectPerformance(**item)
        for item in performance_data.get("subject_performance", [])
    ]
    materials = [
        StudyMaterial(**item)
        for item in materials_data.get("materials", [])
    ]
    tests = [
        UpcomingTest(
            test_id=item["test_id"],
            subject=item["subject"],
            test_name=item["test_name"],
            date=date.fromisoformat(item["date"]),
            topics=list(item["topics"]),
        )
        for item in tests_data.get("upcoming_tests", [])
    ]

    return StudentData(
        profile=profile,
        performance=performance,
        materials=materials,
        tests=tests,
    )

