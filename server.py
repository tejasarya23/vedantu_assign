from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from study_assistant.assistant import StudyAssistant
from study_assistant.data_loader import DEFAULT_DATA_DIR
from study_assistant.langgraph_agent import StudyAgentRunner
from study_assistant.serialization import response_to_dict


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    today: date | None = None


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    model: str


def create_app(
    data_dir: str | Path | None = None,
    *,
    agent: StudyAgentRunner | None = None,
) -> FastAPI:
    assistant = (
        StudyAssistant.from_data_dir(data_dir, agent=agent)
        if data_dir
        else StudyAssistant.from_data_dir(
            os.environ.get("STUDY_ASSISTANT_DATA_DIR", DEFAULT_DATA_DIR),
            agent=agent,
        )
    )
    app = FastAPI(
        title="Student Study Assistant API",
        version="0.1.0",
        description="Personalized study recommendations using LangGraph ReAct, semantic retrieval, and OpenAI tool calling.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "model": os.environ.get("OPENAI_MODEL", "gpt-5.6"),
            "agent": "langgraph_react",
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            response = assistant.answer(
                request.query.strip(),
                today=request.today,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        payload = response_to_dict(response)
        payload["model"] = os.environ.get("OPENAI_MODEL", "gpt-5.6")
        return payload

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastAPI study assistant backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "study_assistant.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
