from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    EventLogRow,
    GameState,
    GameSummary,
    RichGameSummary,
)


class JsonlLogger:
    """Writes per-event JSONL logs and game summary JSON for one game."""

    def __init__(
        self,
        outdir: str,
        game_id: str,
        *,
        filename: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.outdir = Path(outdir)
        self.game_id = game_id
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.outdir.mkdir(parents=True, exist_ok=True)
        log_filename = filename or f"{game_id}.jsonl"
        self.log_path = self.outdir / log_filename
        self._file = self.log_path.open("a", encoding="utf-8")

    def log(self, row: EventLogRow) -> None:
        self._file.write(row.model_dump_json() + "\n")
        self._file.flush()

    def log_event(
        self,
        *,
        game_id: str,
        seed: int,
        condition: str,
        round_idx: int,
        phase: str,
        actor_id: int,
        action_type: str,
        payload: Dict[str, Any],
        actor_role: Optional[str] = None,
    ) -> None:
        row = EventLogRow(
            game_id=game_id,
            seed=seed,
            condition=condition,
            round=round_idx,
            phase=phase,
            actor_id=actor_id,
            action_type=action_type,
            payload=payload,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            actor_role=actor_role,
        )
        self.log(row)

    def write_summary(
        self, state: GameState | Dict[str, Any], extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Write a basic game summary (used by the original run-one / run-batch commands)."""
        if isinstance(state, dict):
            normalized_state = GameState.model_validate(state)
        else:
            normalized_state = state

        summary = GameSummary(
            game_id=normalized_state.game_id,
            seed=normalized_state.config.seed,
            condition=normalized_state.config.condition_name,
            winner=normalized_state.winner,
            rounds=normalized_state.round_idx,
            eliminated_order=normalized_state.eliminated_order,
            config=normalized_state.config,
            roles=normalized_state.roles,
        ).model_dump(mode="json")
        if extra:
            summary.update(extra)
        summary_path = self.outdir / f"{normalized_state.game_id}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return str(summary_path)

    def write_rich_summary(self, rich_summary: RichGameSummary) -> str:
        """Write a full RichGameSummary JSON (used by Experiment 1 commands)."""
        summary_path = self.outdir / "game_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(rich_summary.model_dump(mode="json"), handle, indent=2)
        return str(summary_path)

    def read_events(self) -> List[Dict[str, Any]]:
        """Return all logged events as a list of dicts (reads the JSONL file)."""
        events: List[Dict[str, Any]] = []
        if not self.log_path.exists():
            return events
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    events.append(json.loads(stripped))
        return events

    def close(self) -> None:
        self._file.close()


class ExperimentOutputManager:
    """Manages the directory structure and aggregate output files for an experiment run.

    Directory layout::

        <base_outdir>/
          experiment_1_baseline_behaviour/
            run_<run_id>/
              manifest.json
              summary.csv
              summary.json
              per_game_metrics.csv
              per_round_metrics.csv
              per_agent_metrics.csv
              games/
                <game_id>/
                  events.jsonl
                  game_summary.json
    """

    EXPERIMENT_NAME = "experiment_1_baseline_behaviour"

    def __init__(self, base_outdir: str, run_id: str) -> None:
        self.run_dir = Path(base_outdir) / self.EXPERIMENT_NAME / f"run_{run_id}"
        self.games_dir = self.run_dir / "games"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.games_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self.run_dir.name.removeprefix("run_")

    def game_logger(self, game_id: str, model_name: str) -> JsonlLogger:
        """Return a JsonlLogger scoped to this game's subdirectory."""
        game_dir = self.games_dir / game_id
        game_dir.mkdir(parents=True, exist_ok=True)
        return JsonlLogger(
            str(game_dir),
            game_id,
            filename="events.jsonl",
            experiment_name=self.EXPERIMENT_NAME,
            model_name=model_name,
        )

    def write_manifest(self, manifest_data: Dict[str, Any]) -> str:
        path = self.run_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        return str(path)

    def write_csv(self, filename: str, rows: List[Dict[str, Any]]) -> str:
        """Write a list of dicts to a CSV file in the run directory."""
        path = self.run_dir / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            return str(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def append_csv_row(self, filename: str, row: Dict[str, Any]) -> None:
        """Append a single row to a CSV file, writing the header if the file is new."""
        path = self.run_dir / filename
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def write_json(self, filename: str, data: Any) -> str:
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(path)

    @staticmethod
    def make_run_id() -> str:
        """Generate a timestamp-based run ID."""
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
