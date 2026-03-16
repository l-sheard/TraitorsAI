from __future__ import annotations

import json
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from .agent import TraitorsAgent
from .analysis import (
    aggregate_experiment_metrics,
    compute_agent_metrics,
    compute_game_metrics,
    compute_round_metrics,
)
from .config import create_llm, load_env
from .game_engine import assign_roles, generate_game_id
from .graph import build_graph
from .logging_utils import ExperimentOutputManager, JsonlLogger
from .personas import assign_personas
from .schemas import (
    AgentPrivateState,
    ExperimentManifest,
    GameConfig,
    GameState,
    RichGameSummary,
)

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Experiment 1 constants
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "experiment_1_baseline_behaviour"
EXPERIMENT_CONDITION = "baseline_memory"

METRIC_DEFINITIONS: Dict[str, str] = {
    "banishment_accuracy": "Fraction of banished players who were traitors.",
    "deception_success_rate": "Fraction of rounds (with traitor alive) where the banished player was faithful.",
    "belief_action_alignment_top1": "Fraction of votes where the target was the agent's single most-suspicious player.",
    "belief_action_alignment_top2": "Fraction of votes where the target was in the agent's top-2 most-suspicious players.",
    "suspicion_gap": "Mean suspicion assigned to traitors minus mean suspicion assigned to faithful (from faithful agents).",
    "traitor_vote_agreement_rate": "Fraction of voting rounds (≥2 alive traitors) where all traitors voted for the same target.",
    "murder_vote_agreement_rate": "Fraction of murder rounds (≥2 alive traitors) where all traitors chose the same murder target.",
    "accusation_rate": "Fraction of public messages containing a player reference and an accusation keyword (suspect/suspicious/traitor/lying/liar/untrustworthy).",
    "defence_rate": "Fraction of public messages containing a player reference and a defence keyword (trust/innocent/faithful/defend/clear/vouch/agree with).",
}

# ---------------------------------------------------------------------------
# Internal helpers (shared by old and new commands)
# ---------------------------------------------------------------------------


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
        action_type="game_start",
        payload={
            "n_players": state.config.n_players,
            "n_traitors": state.config.n_traitors,
            "condition": state.config.condition_name,
            "model_name": state.config.model_name,
            "seed": state.config.seed,
        },
    )
    logger.log_event(
        game_id=state.game_id,
        seed=state.config.seed,
        condition=state.config.condition_name,
        round_idx=0,
        phase="setup",
        actor_id=-1,
        action_type="assign_roles",
        payload={"roles": {str(pid): role.value for pid, role in state.roles.items()}},
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
            actor_role=state.roles[pid].value,
        )


def _coerce_final_state(raw_state: GameState | dict) -> GameState:
    if isinstance(raw_state, GameState):
        return raw_state
    return GameState.model_validate(raw_state)


def _run_single_game(config: GameConfig, outdir: str) -> GameState:
    state = _init_game_state(config)
    typer.echo(f"\n\U0001f3ae Starting game: {state.game_id}")
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
        extra={"personas": personas_by_player},
    )
    logger.close()
    typer.echo(f"\n\u2705 Game complete! Winner: {final_state.winner} after {final_state.round_idx} rounds")
    typer.echo(f"   Logs: {log_dir / f'{state.game_id}.jsonl'}\n")
    return final_state


# ---------------------------------------------------------------------------
# Experiment 1: single-game run helper (used by both exp1 commands)
# ---------------------------------------------------------------------------


def _run_experiment_1_game(
    config: GameConfig,
    output_manager: ExperimentOutputManager,
) -> Optional[Dict[str, object]]:
    """Run one game under Experiment 1 conditions.

    Returns a flat per-game metrics dict on success, or None on failure
    (failure details are already written to the game directory).
    """
    state = _init_game_state(config)
    game_id = state.game_id
    typer.echo(f"  \U0001f3ae Game {game_id} (seed={config.seed})")

    logger = output_manager.game_logger(game_id, config.model_name)
    agents, personas_by_player = _build_agents(config, state)
    _log_game_setup(logger, state, personas_by_player)

    try:
        graph = build_graph(agents, logger)
        final_state = _coerce_final_state(graph.invoke(state))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        typer.echo(f"  \u274c ERROR in game {game_id}: {exc}")
        error_path = output_manager.games_dir / game_id / "error.json"
        error_path.write_text(
            json.dumps({"game_id": game_id, "error": str(exc), "traceback": tb}, indent=2),
            encoding="utf-8",
        )
        logger.close()
        return None

    # Build the rich game summary
    traitors_list = sorted(final_state.traitors)
    faithful_list = sorted({pid for pid in final_state.roles if final_state.roles[pid].value == "faithful"})
    final_alive = sorted(final_state.alive)
    final_traitors_alive = sorted(final_state.alive & final_state.traitors)
    final_faithful_alive = sorted(final_state.alive - final_state.traitors)

    events = logger.read_events()

    rich_summary_base = {
        "game_id": game_id,
        "experiment_name": EXPERIMENT_NAME,
        "condition": config.condition_name,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "model_name": config.model_name,
        "temperature": config.temperature,
        "personas": {str(k): v for k, v in personas_by_player.items()},
        "roles": {str(pid): role.value for pid, role in final_state.roles.items()},
        "winner": final_state.winner,
        "faithful_win": final_state.winner == "faithful",
        "traitor_win": final_state.winner == "traitors",
        "total_rounds": final_state.round_idx,
        "eliminated_order": final_state.eliminated_order,
        "final_alive": final_alive,
        "final_traitors_alive": final_traitors_alive,
        "final_faithful_alive": final_faithful_alive,
        # Failure counters
        "structured_output_parse_failures_count": final_state.parse_failure_count,
        "vote_fallback_count": final_state.vote_fallback_count,
        "murder_fallback_count": final_state.murder_fallback_count,
        "belief_update_fallback_count": final_state.belief_update_fallback_count,
        "total_llm_errors_count": final_state.llm_error_count,
        "retries_used_count": final_state.retry_count,
    }

    # Compute derived metrics and merge
    game_metrics = compute_game_metrics(events, rich_summary_base)
    rich_summary_base.update(game_metrics)

    rich_summary = RichGameSummary(**{
        k: rich_summary_base[k]
        for k in RichGameSummary.model_fields
        if k in rich_summary_base
    })
    logger.write_rich_summary(rich_summary)
    logger.close()

    typer.echo(f"     winner={final_state.winner}, rounds={final_state.round_idx}")
    return game_metrics


# ---------------------------------------------------------------------------
# Original CLI commands (preserved)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Experiment 1 CLI commands
# ---------------------------------------------------------------------------


@app.command("experiment-1-run-one")
def experiment_1_run_one(
    seed: int = typer.Option(1, help="Random seed"),
    model_name: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    temperature: float = typer.Option(0.3, help="LLM temperature"),
    n_players: int = typer.Option(9, help="Number of players"),
    n_traitors: int = typer.Option(2, help="Number of traitors"),
    discussion_turns: int = typer.Option(1, help="Discussion turns per round"),
    max_rounds: int = typer.Option(30, help="Maximum rounds"),
    outdir: str = typer.Option("results", help="Base output directory"),
) -> None:
    """Run a single Experiment 1 baseline_memory game."""
    load_env()
    config = GameConfig(
        seed=seed,
        condition_name=EXPERIMENT_CONDITION,
        model_name=model_name,
        temperature=temperature,
        n_players=n_players,
        n_traitors=n_traitors,
        discussion_turns=discussion_turns,
        max_rounds=max_rounds,
    )
    run_id = ExperimentOutputManager.make_run_id()
    output_manager = ExperimentOutputManager(outdir, run_id)

    typer.echo(f"\n\U0001f9ea Experiment 1: Baseline Behaviour")
    typer.echo(f"Run ID : {run_id}")
    typer.echo(f"Output : {output_manager.run_dir}\n")

    seed_list = [seed]
    manifest = ExperimentManifest(
        experiment_name=EXPERIMENT_NAME,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        temperature=temperature,
        seed_list=seed_list,
        number_of_games=1,
        fixed_config=config.model_dump(mode="json"),
        metric_definitions=METRIC_DEFINITIONS,
    )
    output_manager.write_manifest(manifest.model_dump(mode="json"))

    game_metrics = _run_experiment_1_game(config, output_manager)
    if game_metrics is None:
        typer.echo("\n\u274c Game failed. See error.json in the game directory.")
        raise typer.Exit(code=1)

    # Write aggregate outputs (single game)
    _write_experiment_outputs(
        output_manager=output_manager,
        per_game_rows=[game_metrics],
        seed_list=seed_list,
        run_id=run_id,
    )

    typer.echo(f"\n\u2705 Experiment 1 complete. Results: {output_manager.run_dir}")


@app.command("experiment-1-run-batch")
def experiment_1_run_batch(
    seeds: str = typer.Option("1..5", help="Seed range like '1..100' or a single int"),
    model_name: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    temperature: float = typer.Option(0.3, help="LLM temperature"),
    n_players: int = typer.Option(9, help="Number of players"),
    n_traitors: int = typer.Option(2, help="Number of traitors"),
    discussion_turns: int = typer.Option(1, help="Discussion turns per round"),
    max_rounds: int = typer.Option(30, help="Maximum rounds"),
    outdir: str = typer.Option("results", help="Base output directory"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Abort on first game failure"),
) -> None:
    """Run a batch of Experiment 1 baseline_memory games across many seeds."""
    load_env()
    seed_list = _parse_seeds(seeds)
    run_id = ExperimentOutputManager.make_run_id()
    output_manager = ExperimentOutputManager(outdir, run_id)

    base_config = GameConfig(
        seed=seed_list[0],
        condition_name=EXPERIMENT_CONDITION,
        model_name=model_name,
        temperature=temperature,
        n_players=n_players,
        n_traitors=n_traitors,
        discussion_turns=discussion_turns,
        max_rounds=max_rounds,
    )
    manifest = ExperimentManifest(
        experiment_name=EXPERIMENT_NAME,
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        temperature=temperature,
        seed_list=seed_list,
        number_of_games=len(seed_list),
        fixed_config=base_config.model_dump(mode="json"),
        metric_definitions=METRIC_DEFINITIONS,
    )
    output_manager.write_manifest(manifest.model_dump(mode="json"))

    typer.echo(f"\n\U0001f9ea Experiment 1: Baseline Behaviour — Batch Run")
    typer.echo(f"Run ID  : {run_id}")
    typer.echo(f"Seeds   : {seed_list[0]}..{seed_list[-1]} ({len(seed_list)} games)")
    typer.echo(f"Model   : {model_name} (temp={temperature})")
    typer.echo(f"Output  : {output_manager.run_dir}\n")

    per_game_rows: List[Dict[str, object]] = []
    failed = 0
    for i, seed in enumerate(seed_list, 1):
        typer.echo(f"[{i}/{len(seed_list)}] seed={seed}")
        config = GameConfig(
            seed=seed,
            condition_name=EXPERIMENT_CONDITION,
            model_name=model_name,
            temperature=temperature,
            n_players=n_players,
            n_traitors=n_traitors,
            discussion_turns=discussion_turns,
            max_rounds=max_rounds,
        )
        result = _run_experiment_1_game(config, output_manager)
        if result is None:
            failed += 1
            if fail_fast:
                typer.echo("\n\u274c --fail-fast: aborting after first failure.")
                raise typer.Exit(code=1)
        else:
            per_game_rows.append(result)
            output_manager.append_csv_row("per_game_metrics.csv", result)

    if not per_game_rows:
        typer.echo("\n\u274c All games failed. No aggregate outputs written.")
        raise typer.Exit(code=1)

    _write_experiment_outputs(
        output_manager=output_manager,
        per_game_rows=per_game_rows,
        seed_list=seed_list,
        run_id=run_id,
    )

    typer.echo(f"\n\u2705 Batch complete. {len(per_game_rows)} succeeded, {failed} failed.")
    typer.echo(f"   Results: {output_manager.run_dir}")

    # Print summary to console
    agg = aggregate_experiment_metrics(per_game_rows, EXPERIMENT_NAME, run_id)
    typer.echo("\n--- Aggregate Metrics ---")
    for k, v in agg.items():
        if k not in ("experiment_name", "run_id") and v is not None:
            typer.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Shared output writer
# ---------------------------------------------------------------------------


def _rebuild_csvs_from_games_dir(games_dir: Path, run_id: str, output_manager: ExperimentOutputManager) -> int:
    """Scan games_dir for saved game files and rebuild all experiment CSVs.

    Returns the number of games successfully processed.
    """
    per_game_rows: List[Dict[str, object]] = []
    all_round_rows: List[Dict[str, object]] = []
    all_agent_rows: List[Dict[str, object]] = []

    game_dirs = sorted(d for d in games_dir.iterdir() if d.is_dir())
    for game_dir in game_dirs:
        events_path = game_dir / "events.jsonl"
        summary_path = game_dir / "game_summary.json"
        if not summary_path.exists():
            typer.echo(f"  Skipping {game_dir.name}: no game_summary.json")
            continue
        events: List[Dict[str, object]] = []
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        events.append(json.loads(stripped))
        with summary_path.open("r", encoding="utf-8") as f:
            game_summary = json.load(f)
        try:
            game_metrics = compute_game_metrics(events, game_summary)
            per_game_rows.append(game_metrics)
            all_round_rows.extend(compute_round_metrics(events, game_summary))
            all_agent_rows.extend(compute_agent_metrics(events, game_summary))
            typer.echo(f"  Loaded {game_summary.get('game_id', game_dir.name)}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  Warning: failed to process {game_dir.name}: {exc}")

    if not per_game_rows:
        return 0

    output_manager.write_csv("per_game_metrics.csv", per_game_rows)
    output_manager.write_csv("per_round_metrics.csv", all_round_rows)
    output_manager.write_csv("per_agent_metrics.csv", all_agent_rows)
    agg = aggregate_experiment_metrics(per_game_rows, EXPERIMENT_NAME, run_id)
    output_manager.write_csv("summary.csv", [agg])
    output_manager.write_json("summary.json", agg)

    return len(per_game_rows)


@app.command("rebuild-experiment-1-csvs")
def rebuild_experiment_1_csvs(
    run_dir: str = typer.Argument(..., help="Path to the run directory (e.g. results/experiment_1_baseline_behaviour/run_<id>)"),
) -> None:
    """Rebuild all experiment CSVs from saved game files in a (possibly partial) run directory.

    Use this to regenerate per_game_metrics.csv, per_round_metrics.csv,
    per_agent_metrics.csv, summary.csv and summary.json after an interrupted batch run,
    so you can then run the analysis pipeline on the saved games.
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        typer.echo(f"Error: run directory not found: {run_path}")
        raise typer.Exit(code=1)
    games_dir = run_path / "games"
    if not games_dir.exists():
        typer.echo(f"Error: games/ subdirectory not found in {run_path}")
        raise typer.Exit(code=1)

    run_id = run_path.name.removeprefix("run_")
    # Reconstruct output_manager pointing at the existing run dir by passing parent dirs
    base_outdir = str(run_path.parent.parent)
    output_manager = ExperimentOutputManager(base_outdir, run_id)

    typer.echo(f"\n\U0001f527 Rebuilding CSVs from saved game files")
    typer.echo(f"Run dir : {run_path}")

    n = _rebuild_csvs_from_games_dir(games_dir, run_id, output_manager)
    if n == 0:
        typer.echo("\n\u274c No valid game files found. Nothing written.")
        raise typer.Exit(code=1)

    typer.echo(f"\n\u2705 Rebuilt CSVs from {n} game(s). Run the analysis pipeline to generate graphs:")
    typer.echo(f"   python -m traitors_ai.analysis analyse-experiment-1 --run-dir {run_path}")


# ---------------------------------------------------------------------------
# Shared output writer
# ---------------------------------------------------------------------------


def _write_experiment_outputs(
    output_manager: ExperimentOutputManager,
    per_game_rows: List[Dict[str, object]],
    seed_list: List[int],
    run_id: str,
) -> None:
    """Write per_game_metrics.csv, per_round_metrics.csv, per_agent_metrics.csv,
    summary.csv and summary.json for the completed batch."""

    # per_game_metrics.csv (overwrite the incrementally-written file with final consistent data)
    output_manager.write_csv("per_game_metrics.csv", per_game_rows)

    # per_round_metrics.csv and per_agent_metrics.csv – load from saved game summaries
    all_round_rows: List[Dict[str, object]] = []
    all_agent_rows: List[Dict[str, object]] = []

    for row in per_game_rows:
        game_id = row["game_id"]
        game_dir = output_manager.games_dir / game_id
        events_path = game_dir / "events.jsonl"
        summary_path = game_dir / "game_summary.json"
        if not events_path.exists() or not summary_path.exists():
            continue
        import json as _json
        events = []
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    events.append(_json.loads(stripped))
        with summary_path.open("r", encoding="utf-8") as f:
            game_summary = _json.load(f)
        all_round_rows.extend(compute_round_metrics(events, game_summary))
        all_agent_rows.extend(compute_agent_metrics(events, game_summary))

    output_manager.write_csv("per_round_metrics.csv", all_round_rows)
    output_manager.write_csv("per_agent_metrics.csv", all_agent_rows)

    # Aggregate summary
    agg = aggregate_experiment_metrics(per_game_rows, EXPERIMENT_NAME, run_id)
    output_manager.write_csv("summary.csv", [agg])
    output_manager.write_json("summary.json", agg)


if __name__ == "__main__":
    app()
