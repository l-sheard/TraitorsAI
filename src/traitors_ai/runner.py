from __future__ import annotations

import json
import os
import random
from typing import List, Optional

import typer

from .agent import TraitorsAgent
from .config import create_llm, load_env
from .game_engine import assign_roles, generate_game_id
from .graph import build_graph
from .logging_utils import JsonlLogger
from .personas import assign_personas
from .schemas import AgentPrivateState, GameConfig, GameState

app = typer.Typer(add_completion=False)


def _parse_seeds(seed_arg: str) -> List[int]:
    if ".." in seed_arg:
        start, end = seed_arg.split("..")
        return list(range(int(start), int(end) + 1))
    return [int(seed_arg)]


def _init_game_state(config: GameConfig) -> GameState:
    rng = random.Random(config.seed)
    roles, traitors = assign_roles(config.n_players, config.n_traitors, rng)
    game_id = generate_game_id(config.seed, config.condition_name)
    alive = set(range(1, config.n_players + 1))
    agent_states = {
        pid: AgentPrivateState(
            memory_summary="" if config.condition_name == "no_memory" else "",
            suspicion_scores={
                other: 0.5 for other in alive if other != pid
            },
        )
        for pid in alive
    }
    return GameState(
        config=config,
        game_id=game_id,
        round_idx=1,
        alive=alive,
        roles=roles,
        traitors=traitors,
        public_transcript=[],
        vote_history=[],
        traitor_private_transcript=[],
        agent_states=agent_states,
        rng=rng,
    )


def _build_agents(config: GameConfig, state: GameState) -> dict[int, TraitorsAgent]:
    rng = random.Random(config.seed)
    personas = assign_personas(config.n_players, rng)
    llm = create_llm(config.model_name, config.temperature)
    agents = {}
    for pid in range(1, config.n_players + 1):
        agents[pid] = TraitorsAgent(
            agent_id=pid,
            persona=personas[pid - 1],
            role=state.roles[pid].value,
            llm_client=llm,
            config=config,
        )
    return agents


def _run_single_game(config: GameConfig, outdir: str) -> GameState:
    state = _init_game_state(config)
    print(f"\n🎮 Starting game: {state.game_id}")
    print(f"   Players: {config.n_players} ({config.n_traitors} traitors)")
    print(f"   Seed: {config.seed}, Condition: {config.condition_name}\n")
    log_dir = os.path.join(outdir, "logs")
    logger = JsonlLogger(log_dir, state.game_id)
    agents = _build_agents(config, state)
    graph = build_graph(agents, logger)
    final_state = graph.invoke(state)
    logger.write_summary(final_state)
    logger.close()
    # Handle dict return from LangGraph
    winner = final_state["winner"] if isinstance(final_state, dict) else final_state.winner
    round_idx = final_state["round_idx"] if isinstance(final_state, dict) else final_state.round_idx
    print(f"\n✅ Game complete! Winner: {winner} after {round_idx} rounds")
    print(f"   Logs: {log_dir}/{state.game_id}.jsonl\n")
    return final_state


@app.command("run-one")
def run_one(
    seed: int = typer.Option(1, help="Random seed"),
    condition: str = typer.Option("baseline_memory", help="Condition name"),
    model_name: str = typer.Option("gpt-4o-mini", help="Model name"),
    temperature: float = typer.Option(0.3, help="Temperature"),
    n_players: int = typer.Option(9, help="Number of players"),
    n_traitors: int = typer.Option(2, help="Number of traitors"),
    discussion_turns: int = typer.Option(1, help="Discussion turns per round"),
    max_rounds: int = typer.Option(30, help="Maximum rounds"),
    outdir: str = typer.Option("results", help="Output directory"),
) -> None:
    load_env()
    config = GameConfig(
        seed=seed,
        condition_name=condition,
        model_name=model_name,
        temperature=temperature,
        n_players=n_players,
        n_traitors=n_traitors,
        discussion_turns=discussion_turns,
        max_rounds=max_rounds,
    )
    state = _run_single_game(config, outdir)
    # Handle dict return from LangGraph
    game_id = state["game_id"] if isinstance(state, dict) else state.game_id
    winner = state["winner"] if isinstance(state, dict) else state.winner
    round_idx = state["round_idx"] if isinstance(state, dict) else state.round_idx
    typer.echo(json.dumps({
        "game_id": game_id,
        "winner": winner,
        "rounds": round_idx,
    }))


@app.command("run-batch")
def run_batch(
    seeds: str = typer.Option("1..5", help="Seed range like 1..100"),
    condition: str = typer.Option("baseline_memory", help="Condition name"),
    model_name: str = typer.Option("gpt-4o-mini", help="Model name"),
    temperature: float = typer.Option(0.3, help="Temperature"),
    n_players: int = typer.Option(9, help="Number of players"),
    n_traitors: int = typer.Option(2, help="Number of traitors"),
    discussion_turns: int = typer.Option(1, help="Discussion turns per round"),
    max_rounds: int = typer.Option(30, help="Maximum rounds"),
    outdir: str = typer.Option("results", help="Output directory"),
) -> None:
    load_env()
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for seed in _parse_seeds(seeds):
        config = GameConfig(
            seed=seed,
            condition_name=condition,
            model_name=model_name,
            temperature=temperature,
            n_players=n_players,
            n_traitors=n_traitors,
            discussion_turns=discussion_turns,
            max_rounds=max_rounds,
        )
        state = _run_single_game(config, outdir)
        rows.append(
            {
                "game_id": state.game_id,
                "seed": seed,
                "condition": condition,
                "winner": state.winner,
                "rounds": state.round_idx,
                "traitor_win": state.winner == "traitors",
                "faithful_win": state.winner == "faithful",
            }
        )
    summary_path = os.path.join(outdir, "summary.csv")
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
    except Exception:
        import csv

        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    typer.echo(f"Wrote {summary_path}")


if __name__ == "__main__":
    app()
