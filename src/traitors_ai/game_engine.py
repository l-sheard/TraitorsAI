from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple

from .schemas import Role


def assign_roles(n_players: int, n_traitors: int, rng: random.Random) -> Tuple[Dict[int, Role], Set[int]]:
    player_ids = list(range(1, n_players + 1))
    traitors = set(rng.sample(player_ids, n_traitors))
    roles = {pid: (Role.traitor if pid in traitors else Role.faithful) for pid in player_ids}
    return roles, traitors


def apply_vote(
    alive: Set[int],
    votes: Dict[int, int],
    rng: random.Random,
) -> Tuple[Optional[int], Dict[str, object]]:
    counts: Dict[int, int] = {pid: 0 for pid in alive}
    for target in votes.values():
        if target in counts:
            counts[target] += 1
    max_votes = max(counts.values()) if counts else 0
    top = [pid for pid, cnt in counts.items() if cnt == max_votes and cnt > 0]
    if len(top) == 1:
        return top[0], {"tied": [], "counts": counts, "random": False}
    if len(top) > 1:
        return None, {"tied": top, "counts": counts, "random": False}
    if not alive:
        return None, {"tied": [], "counts": counts, "random": False}
    choice = rng.choice(sorted(alive))
    return choice, {"tied": list(alive), "counts": counts, "random": True}


def apply_murder(
    alive: Set[int],
    traitors: Set[int],
    murder_votes: Dict[int, int],
    rng: random.Random,
) -> Optional[int]:
    faithful_alive = sorted(alive - traitors)
    if not faithful_alive:
        return None
    counts: Dict[int, int] = {pid: 0 for pid in faithful_alive}
    for target in murder_votes.values():
        if target in counts:
            counts[target] += 1
    max_votes = max(counts.values()) if counts else 0
    top = [pid for pid, cnt in counts.items() if cnt == max_votes and cnt > 0]
    if len(top) == 1:
        return top[0]
    if len(top) > 1:
        return rng.choice(sorted(top))
    return rng.choice(faithful_alive)


def check_terminal(alive: Set[int], traitors: Set[int]) -> Optional[str]:
    if not traitors:
        return "faithful"
    faithful_count = len(alive) - len(traitors)
    if len(traitors) >= faithful_count:
        return "traitors"
    return None


def generate_game_id(seed: int, condition: str) -> str:
    digest = hashlib.sha256(f"{seed}-{condition}".encode("utf-8")).hexdigest()[:8]
    return f"{condition}-{seed}-{digest}"
