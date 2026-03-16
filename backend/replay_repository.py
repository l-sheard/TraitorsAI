from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ReplayRepository:
    """Small data access layer for replay summaries and event logs."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir

    def list_games(self) -> List[Dict[str, Any]]:
        if not self.logs_dir.exists():
            return []

        games: List[Dict[str, Any]] = []
        # Legacy flat logs format: <game_id>_summary.json + <game_id>.jsonl
        for file_path in sorted(self.logs_dir.glob("*_summary.json"), reverse=True):
            try:
                game = self._read_json(file_path)
                if "game_id" in game:
                    game["replay_id"] = str(game["game_id"])
                games.append(game)
            except (OSError, json.JSONDecodeError):
                continue

        # Run-directory format:
        # results/experiment_1_baseline_behaviour/run_<id>/games/<game_id>/game_summary.json
        run_root = self.logs_dir / "experiment_1_baseline_behaviour"
        if run_root.exists():
            for run_dir in sorted(run_root.glob("run_*"), reverse=True):
                games_dir = run_dir / "games"
                if not games_dir.exists():
                    continue
                for game_dir in sorted((d for d in games_dir.iterdir() if d.is_dir()), reverse=True):
                    summary_path = game_dir / "game_summary.json"
                    if not summary_path.exists():
                        continue
                    try:
                        game = self._read_json(summary_path)
                        game_id = str(game.get("game_id", game_dir.name))
                        game["game_id"] = game_id
                        game["run_id"] = run_dir.name
                        game["replay_id"] = f"{run_dir.name}::{game_id}"
                        games.append(game)
                    except (OSError, json.JSONDecodeError):
                        continue

        games.sort(
            key=lambda game: (
                game.get("run_id", ""),
                game.get("seed", 0),
                game.get("game_id", ""),
            ),
            reverse=True,
        )
        return games

    def get_summary(self, game_id: str) -> Dict[str, Any]:
        summary_path, _ = self._resolve_paths(game_id)
        if not summary_path.exists():
            raise FileNotFoundError(f"Game {game_id} not found")
        game = self._read_json(summary_path)
        if "::" in game_id:
            run_id, parsed_game_id = self._split_replay_id(game_id)
            game["run_id"] = run_id
            game["game_id"] = parsed_game_id
            game["replay_id"] = game_id
        else:
            game["replay_id"] = game_id
        return game

    def get_events(self, game_id: str) -> List[Dict[str, Any]]:
        _, log_path = self._resolve_paths(game_id)
        if not log_path.exists():
            raise FileNotFoundError(f"Game log {game_id} not found")

        events: List[Dict[str, Any]] = []
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    events.append(json.loads(stripped))
        return events

    def get_personas(self, game_id: str) -> Dict[int, Dict[str, Any]]:
        summary = self.get_summary(game_id)
        personas = summary.get("personas")
        if isinstance(personas, dict) and personas:
            return {int(player_id): persona for player_id, persona in personas.items()}

        extracted: Dict[int, Dict[str, Any]] = {}
        for event in self.get_events(game_id):
            if event.get("action_type") == "assign_persona":
                player_id = event.get("actor_id")
                if isinstance(player_id, int):
                    extracted[player_id] = event.get("payload", {}).get("persona", {})
        return extracted

    def _resolve_paths(self, replay_id: str) -> Tuple[Path, Path]:
        # Run-directory replay id format: run_<id>::<game_id>
        if "::" in replay_id:
            run_id, game_id = self._split_replay_id(replay_id)
            game_dir = self.logs_dir / "experiment_1_baseline_behaviour" / run_id / "games" / game_id
            return game_dir / "game_summary.json", game_dir / "events.jsonl"

        # Legacy flat logs format
        return self.logs_dir / f"{replay_id}_summary.json", self.logs_dir / f"{replay_id}.jsonl"

    @staticmethod
    def _split_replay_id(replay_id: str) -> Tuple[str, str]:
        if "::" not in replay_id:
            raise ValueError(f"Invalid replay id: {replay_id}")
        run_id, game_id = replay_id.split("::", 1)
        return run_id, game_id

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
