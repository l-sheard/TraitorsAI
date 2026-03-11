from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import EventLogRow, GameState, GameSummary


class JsonlLogger:
    def __init__(self, outdir: str, game_id: str) -> None:
        self.outdir = Path(outdir)
        self.game_id = game_id
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.outdir / f"{game_id}.jsonl"
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
        )
        self.log(row)

    def write_summary(self, state: GameState | Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
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

    def close(self) -> None:
        self._file.close()
