from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
        if int(end) < int(start):
            raise typer.BadParameter("Seed range end must be greater than or equal to the start")
        return list(range(int(start), int(end) + 1))
    return [int(seed_arg)]


def _init_game_state(config: GameConfig) -> GameState:
    rng = random.Random(config.seed)
    roles, traitors = assign_roles(config.n_players, config.n_traitors, rng)
    game_id = generate_game_id(config.seed, config.condition_name)
    alive = set(range(1, config.n_players + 1))
    agent_states = {
        pid: AgentPrivateState(
            memory_summary="",
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


def _build_agents(config: GameConfig, state: GameState) -> Tuple[Dict[int, TraitorsAgent], Dict[int, Dict[str, object]]]:
    rng = random.Random(config.seed)
    persona_cards = assign_personas(config.n_players, rng)
    llm = create_llm(config.model_name, config.temperature)
    agents: Dict[int, TraitorsAgent] = {}
    personas_by_player: Dict[int, Dict[str, object]] = {}
    for pid in range(1, config.n_players + 1):
        persona = persona_cards[pid - 1]
        agents[pid] = TraitorsAgent(
            agent_id=pid,
            persona=persona,
            role=state.roles[pid].value,
            llm_client=llm,
            config=config,
        )
        personas_by_player[pid] = persona
    return agents, personas_by_player


def _log_game_setup(
    logger: JsonlLogger,
    state: GameState,
    personas_by_player: Dict[int, Dict[str, object]],
) -> None:
    logger.log_event(
        game_id=state.game_id,
        seed=state.config.seed,
        condition=state.config.condition_name,
        round_idx=0,
        phase="setup",
        actor_id=-1,
        action_type="assign_roles",
        payload={"roles": {pid: role.value for pid, role in state.roles.items()}},
    )
    for pid, persona in personas_by_player.items():
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=0,
            phase="setup",
            actor_id=pid,
            action_type="assign_persona",
            payload={"persona": persona},
        )


def _coerce_final_state(raw_state: GameState | dict) -> GameState:
    if isinstance(raw_state, GameState):
        return raw_state
    return GameState.model_validate(raw_state)


def _run_single_game(config: GameConfig, outdir: str) -> GameState:
    state = _init_game_state(config)
    typer.echo(f"\n🎮 Starting game: {state.game_id}")
    typer.echo(f"   Players: {config.n_players} ({config.n_traitors} traitors)")
    typer.echo(f"   Seed: {config.seed}, Condition: {config.condition_name}\n")
    log_dir = Path(outdir) / "logs"
    logger = JsonlLogger(str(log_dir), state.game_id)
    agents, personas_by_player = _build_agents(config, state)
    _log_game_setup(logger, state, personas_by_player)
    graph = build_graph(agents, logger)
    final_state = _coerce_final_state(graph.invoke(state))
    logger.write_summary(
        final_state,
        extra={
            "personas": personas_by_player,
        },
    )
    logger.close()
    typer.echo(f"\n✅ Game complete! Winner: {final_state.winner} after {final_state.round_idx} rounds")
    typer.echo(f"   Logs: {log_dir / f'{state.game_id}.jsonl'}\n")
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
    typer.echo(json.dumps({
        "game_id": state.game_id,
        "winner": state.winner,
        "rounds": state.round_idx,
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
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    summary_path = output_dir / "summary.csv"
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
    except Exception:
        import csv

        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    typer.echo(f"Wrote {summary_path}")


if __name__ == "__main__":
    app()
