# Traitors AI Simulator

A reproducible, LLM-driven multi-agent social deduction simulator inspired by **The Traitors**. Each player is an autonomous agent with a persona, private memory, and evolving suspicion scores. The rules engine is deterministic given a seed, while LLMs only generate public messages, votes, and traitor murder choices.

## Features
- Deterministic rules engine (seeded RNG)
- LangGraph orchestration
- Persona-based agents
- Strict structured outputs for votes/murders
- JSONL logging per game
- Batch runner with CSV summary
- Max round cutoff with draw outcome

## Setup
1. Create a virtual environment (Python 3.11+).
2. Install dependencies:
   - `pip install -e .`
3. Create a `.env` file based on `.env.example`.
   - Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic`.

## Run a single game
```
python -m traitors_ai.runner run-one --seed 1 --condition baseline_memory
```

## Run batch experiments
```
python -m traitors_ai.runner run-batch --seeds 1..25 --condition baseline_memory --outdir results
```

## Logs & Outputs
- JSONL action logs are stored in `results/logs/{game_id}.jsonl`.
- Game summaries are stored in `results/logs/{game_id}_summary.json`.
- Batch summary CSV is stored in `results/summary.csv`.

## Configuration
You can control:
- Number of players and traitors (`--n-players`, `--n-traitors`)
- Discussion turns (`--discussion-turns`)
- Model name and temperature
- Condition (`baseline_memory` or `no_memory`)

If the game reaches `max_rounds` without a winner, the outcome is recorded as `draw`.

The rules engine is fully deterministic with a fixed seed.

## Notes on reproducibility
- Only the LLM outputs are stochastic; all rule resolution is deterministic.
- For comparable runs, keep model, temperature, and seed fixed.
- Public transcript summaries are truncated to manage token limits.
