from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ReplayRepository:
    """Small data access layer for replay summaries and event logs."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir

    def list_games(self) -> List[Dict[str, Any]]:
        if not self.logs_dir.exists():
            return []

        games: List[Dict[str, Any]] = []
        for file_path in sorted(self.logs_dir.glob("*_summary.json"), reverse=True):
            try:
                games.append(self._read_json(file_path))
            except (OSError, json.JSONDecodeError):
                continue

        games.sort(
            key=lambda game: (
                game.get("seed", 0),
                game.get("game_id", ""),
            ),
            reverse=True,
        )
        return games

    def get_summary(self, game_id: str) -> Dict[str, Any]:
        summary_path = self.logs_dir / f"{game_id}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Game {game_id} not found")
        return self._read_json(summary_path)

    def get_events(self, game_id: str) -> List[Dict[str, Any]]:
        log_path = self.logs_dir / f"{game_id}.jsonl"
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

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
