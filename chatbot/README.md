# Student Study Assistant

A FastAPI + Python AI study assistant that helps a student decide what to study next from profile data, performance history, study materials, and test schedule data.

The app uses a LangGraph ReAct agent with OpenAI. The model automatically calls tools for semantic retrieval, weak topics, test schedule, performance, and study material recommendations before writing the final student-facing answer.

## What It Solves

Example questions:

- `I am weak in Algebra. What should I do next?`
- `What should I study this week?`
- `Which topic should I prioritize first?`
- `I have a Maths test coming up. Help me prepare.`

The assistant responds with:

- a prioritized study topic,
- a daily plan based on available study time,
- relevant materials,
- upcoming or matched test evidence,
- explicit reasoning for why that plan was chosen.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a local environment file:

```bash
cp .env.example .env
```

Set a rotated OpenAI key in `.env` or export it in your shell:

```bash
export OPENAI_API_KEY="your-rotated-key"
export OPENAI_MODEL="gpt-5.6"
```

The pasted key should be rotated before use. Keep the replacement key in your shell or local `.env`; do not commit it.

## Run The CLI

```bash
study-assistant "I am weak in Algebra. What should I do next?" --today 2026-04-10
study-assistant "What should I study this week?" --today 2026-04-10
study-assistant "Which topic should I prioritize first?" --today 2026-04-10
study-assistant "I have a Maths test coming up. Help me prepare." --today 2026-04-10
```

Use JSON output when you want to inspect retrieval and tools:

```bash
study-assistant "I struggle with equation roots" --today 2026-04-10 --json
```

The sample dataset contains a test dated `2026-04-14`. The `--today 2026-04-10` flag keeps the sample test genuinely upcoming for demos and tests.

## Run The API

```bash
study-assistant-api --port 8000
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"I have a Maths test coming up. Help me prepare.","today":"2026-04-10"}'
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

## Data Files

All sample data lives in `data/`:

- `student_profile.json`: student identity, study time, strong and weak topics.
- `performance_history.json`: subject-level scores.
- `study_materials.json`: topic-mapped learning resources.
- `upcoming_tests.json`: test dates, subjects, and covered topics.

## Architecture

- `src/study_assistant/server.py` exposes the FastAPI backend with `/ask`, `/health`, and generated OpenAPI docs.
- `StudyAssistant` in `src/study_assistant/assistant.py` delegates each query to the LangGraph ReAct runner.
- `LangGraphStudyAgent` in `src/study_assistant/langgraph_agent.py` builds the ReAct graph with `create_react_agent`, `ChatOpenAI`, and tool definitions.
- `SemanticSearchIndex` in `src/study_assistant/semantic_search.py` builds searchable documents from structured JSON records.
- `LocalSemanticEmbedder` in `src/study_assistant/embeddings.py` creates deterministic vector embeddings with alias expansion and hashed n-gram features.
- `StudentTools` in `src/study_assistant/tools.py` exposes mandatory tool-style operations:
  - `semantic_search_student_data`
  - `get_weak_topics`
  - `get_upcoming_tests`
  - `get_subject_performance`
  - `recommend_study_material`
- `src/study_assistant/cli.py` provides the command-line interface.

## Retrieval Approach

The assistant combines semantic retrieval with structured filtering.

Semantic embedding search is used where keyword search is brittle:

- `Maths exam` should match a `Mathematics` test.
- `equation roots` should match `Quadratic Equations`.
- `struggle`, `poor`, or `difficult` should match weak-area evidence.
- `ray diagrams`, `optics`, `mirror`, or `lens` should match `Light - Reflection and Refraction`.

Structured filtering is used where exactness matters:

- weak topics come from `student_profile.json`,
- marks come from `performance_history.json`,
- test dates are compared as real dates,
- materials are filtered by known topic IDs/names after semantic ranking.

This split is deliberate: semantic search finds meaning-equivalent context, while tools preserve factual accuracy for student records.

## LangGraph ReAct Flow

The agent uses LangGraph's ReAct loop:

1. The user query enters a LangGraph graph.
2. The model is instructed to call `semantic_search_student_data` first.
3. The model chooses the structured tools it needs, such as `get_weak_topics`, `get_upcoming_tests`, `get_subject_performance`, and `recommend_study_material`.
4. Tool outputs are returned to the graph as evidence.
5. The model writes the final answer from those tool results.

This makes OpenAI the reasoning and response layer while keeping student data access grounded through explicit tools. `OPENAI_API_KEY` is required; if it is missing, the API returns an error instead of falling back to a rule-based final answer.

## Reasoning And Ranking

The assistant ranks study topics using:

1. weak-topic evidence from the student profile,
2. upcoming or matched test topics,
3. low subject performance,
4. semantic relevance to the query.

For Arjun, Mathematics is the lowest-scoring subject at `52%`, so math weak topics receive additional priority. If a near test covers Algebra and Quadratic Equations, those topics become more urgent. The response includes a `Why this recommendation` section and a `Tool calls used` section to make the reasoning auditable.

## Key Decisions

- No frontend: FastAPI and CLI are enough for the assignment and keep the architecture focused.
- LangGraph ReAct agent: the model automatically decides which tools to call inside the graph loop.
- Required OpenAI final generation: retrieval/tools ground the answer, and OpenAI writes the final response.
- Explicit tools: retrieval operations are implemented as named tools instead of hidden helper calls.
- Explainable output: the assistant reports the data and retrieval matches that shaped the answer.
- Date override: `--today` makes stale sample datasets testable and repeatable.

## Limitations

- The local embedding model is not as strong as production embeddings from OpenAI or sentence-transformers.
- The dataset has one student and only a few records.
- Subject-to-topic mapping is simple and currently tuned to the sample CBSE data.
- The API and CLI require a valid `OPENAI_API_KEY`.
- The API currently uses permissive CORS for local development.

## Next Improvements

- Add OpenAI or sentence-transformers embeddings behind the same `LocalSemanticEmbedder` interface.
- Store vectors in FAISS, Chroma, or SQLite vector extensions for larger datasets.
- Add more granular performance history by topic, quiz, date, and mistake type.
- Generate adaptive weekly plans with spaced repetition.
- Add authentication and student selection for multi-student usage.
- Stream LangGraph events from FastAPI so callers can watch tool calls and model tokens live.
- Add a small frontend later only if the product needs one.

## Verification

```bash
pytest
```
