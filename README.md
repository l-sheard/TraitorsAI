# Traitors AI

Traitors AI is an LLM-driven social deduction simulator inspired by **The Traitors**.
It includes a deterministic game engine, a CLI for running simulations, JSONL logging, and a replay viewer.

## Interface Preview

![Traitors AI Replay Console](docs/images/UIscreenshot.png)

## Requirements

- Python 3.11+
- Node.js 18+
- OpenAI or Anthropic API key

## Installation

```bash
pip install -e .[dev]
```

Create `.env` from `.env.example` and set credentials:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

## CLI usage

Run one game:

```bash
python -m traitors_ai.runner run-one --seed 1 --condition baseline_memory
```

Run a batch:

```bash
python -m traitors_ai.runner run-batch --seeds 1..10 --condition baseline_memory --outdir results
```

## Replay viewer

Start backend:

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Start frontend:

```dotenv
cd frontend
npm install
npm start
```

Open http://localhost:3000

## Project structure

- `src/traitors_ai/game_engine.py` — deterministic rules
- `src/traitors_ai/agent.py` — agent behavior and structured LLM parsing
- `src/traitors_ai/graph.py` — simulation flow orchestration
- `src/traitors_ai/runner.py` — CLI entry points
- `backend/app.py` — replay API
- `frontend/src/components/` — replay UI

## Output files

- `results/logs/{game_id}.jsonl` — event log
- `results/logs/{game_id}_summary.json` — game summary
- `results/summary.csv` — batch summary

## Testing

```bash
pytest
```

## Notes

- The rules engine is deterministic for a fixed seed.
- LLM outputs affect discussion, voting, and traitor decisions.
