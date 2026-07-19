from __future__ import annotations

from study_assistant.models import AssistantResponse


def response_to_dict(response: AssistantResponse) -> dict[str, object]:
    return {
        "answer": response.answer,
        "tool_calls": [
            {
                "tool_name": result.tool_name,
                "payload": result.payload,
                "reasoning": result.reasoning,
            }
            for result in response.tool_calls
        ],
        "retrieved_context": [
            {
                "doc_id": result.document.doc_id,
                "kind": result.document.kind,
                "title": result.document.title,
                "text": result.document.text,
                "score": round(result.score, 4),
                "metadata": result.document.metadata,
            }
            for result in response.retrieved_context
        ],
    }

