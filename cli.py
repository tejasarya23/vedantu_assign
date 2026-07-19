from __future__ import annotations

import argparse
import json
from datetime import date

from study_assistant.assistant import StudyAssistant
from study_assistant.serialization import response_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the student study assistant a question.")
    parser.add_argument("query", nargs="*", help="Student question to answer.")
    parser.add_argument("--data-dir", default=None, help="Directory containing JSON data files.")
    parser.add_argument("--today", default=None, help="Override today's date as YYYY-MM-DD.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return answer, retrieved context, and tool calls as JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    query = " ".join(args.query).strip()
    if not query:
        query = input("Ask a study question: ").strip()

    assistant = StudyAssistant.from_data_dir(args.data_dir) if args.data_dir else StudyAssistant()
    today = date.fromisoformat(args.today) if args.today else None
    response = assistant.answer(query, today=today)

    if args.json:
        print(json.dumps(response_to_dict(response), indent=2))
        return

    print(response.answer)


if __name__ == "__main__":
    main()
