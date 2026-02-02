from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .schemas import EventLogRow, GameState


class JsonlLogger:
    def __init__(self, outdir: str, game_id: str) -> None:
        self.outdir = outdir
        self.game_id = game_id
        os.makedirs(outdir, exist_ok=True)
        self.log_path = os.path.join(outdir, f"{game_id}.jsonl")
        self._file = open(self.log_path, "a", encoding="utf-8")

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
        # Handle both GameState objects and dicts (LangGraph returns dicts)
        if isinstance(state, dict):
            game_id = state["game_id"]
            config = state["config"]
            winner = state.get("winner")
            round_idx = state["round_idx"]
            eliminated_order = state.get("eliminated_order", [])
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
        else:
            game_id = state.game_id
            config_dict = state.config.model_dump()
            winner = state.winner
            round_idx = state.round_idx
            eliminated_order = state.eliminated_order
            config = state.config
        
        summary_path = os.path.join(self.outdir, f"{game_id}_summary.json")
        summary = {
            "game_id": game_id,
            "seed": config.seed if hasattr(config, 'seed') else config_dict['seed'],
            "condition": config.condition_name if hasattr(config, 'condition_name') else config_dict['condition_name'],
            "winner": winner,
            "rounds": round_idx,
            "eliminated_order": eliminated_order,
            "config": config_dict,
        }
        if extra:
            summary.update(extra)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary_path

    def close(self) -> None:
        self._file.close()
