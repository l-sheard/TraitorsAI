# Traitors AI

Traitors AI is a reproducible, LLM-driven social deduction simulator inspired by **The Traitors**. The project combines a deterministic game engine, configurable agent orchestration, JSONL replay logging, and a lightweight replay viewer for exploring how autonomous agents reason, speak, vote, and adapt over time.

## Why this repo is useful as a portfolio sample

This codebase is intentionally structured to demonstrate:
- clear Python domain modelling with `pydantic`
- deterministic simulation logic separated from model-driven behaviour
- resilient handling of structured LLM output
- CLI-based experimentation workflows for single runs and batch runs
- replay-oriented backend and frontend layers for observability
- test coverage for core game logic and parsing utilities

## Architecture

### Python simulation package
- `src/traitors_ai/schemas.py` — typed game configuration, state, events, and summaries
- `src/traitors_ai/game_engine.py` — deterministic rules such as role assignment, voting, murder resolution, and terminal checks
- `src/traitors_ai/agent.py` — LLM-backed agent behaviour and structured response handling
- `src/traitors_ai/graph.py` — LangGraph orchestration across discussion, voting, banishment, and murder phases
- `src/traitors_ai/runner.py` — Typer CLI for single simulations and batch experiments
- `src/traitors_ai/logging_utils.py` — JSONL event logging and replay summary generation

### Replay backend
- `backend/app.py` — FastAPI service exposing replay endpoints
- `backend/replay_repository.py` — replay data access layer for summaries, events, and persona metadata

### Replay frontend
- `frontend/src/api/replayApi.js` — shared HTTP client
- `frontend/src/lib/replayState.js` — state reconstruction from replay events
- `frontend/src/components/*` — viewer UI for replay navigation and table visualisation

## Core design choices

### Deterministic engine, stochastic agents
The game engine remains deterministic for a fixed seed. LLMs only influence agent-facing actions such as:
- public discussion messages
- suspicion updates
- votes
- traitor coordination
- murder choices

This keeps experiments reproducible while still allowing agent behaviour to vary with prompt and model choice.

### Structured output normalisation
LLM responses are validated against typed schemas. The project now also normalises common model mistakes, such as returning `"P3"` instead of `3`, which makes replay generation more reliable without weakening validation.

### Replay-first observability
Each game produces:
- a JSONL event stream for step-by-step analysis
- a structured summary JSON for metadata and outcomes
- optional batch CSV summaries for experiment comparison

## Quick start

### 1. Install dependencies
```bash
pip install -e .[dev]
```

### 2. Configure environment
Copy `.env.example` to `.env` and provide API credentials.

Example:
```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

### 3. Run a single simulation
```bash
python -m traitors_ai.runner run-one --seed 1 --condition baseline_memory
```

### 4. Run a batch experiment
```bash
python -m traitors_ai.runner run-batch --seeds 1..10 --condition baseline_memory --outdir results
```

## Replay viewer

### Start the backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

### Start the frontend
```bash
cd frontend
npm install
npm start
```

Then open http://localhost:3000.

## Output files
- `results/logs/{game_id}.jsonl` — replayable event log
- `results/logs/{game_id}_summary.json` — summary metadata and outcome
- `results/summary.csv` — batch experiment overview

## Testing
```bash
pytest
```

## Future improvements
Potential next steps include:
- richer experiment dashboards and aggregate analytics
- better memory strategies beyond summary truncation
- expanded persona generation and calibration
- model benchmarking across conditions and prompt variants
