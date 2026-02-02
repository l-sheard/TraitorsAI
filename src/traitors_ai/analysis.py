from __future__ import annotations

from typing import Dict, List


def summarize_results(rows: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(rows)
    if total == 0:
        return {"total": 0, "traitor_win_rate": 0.0, "faithful_win_rate": 0.0}
    traitor_wins = sum(1 for r in rows if r.get("winner") == "traitors")
    faithful_wins = sum(1 for r in rows if r.get("winner") == "faithful")
    return {
        "total": total,
        "traitor_win_rate": traitor_wins / total,
        "faithful_win_rate": faithful_wins / total,
    }
