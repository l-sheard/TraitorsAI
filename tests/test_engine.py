import random

from traitors_ai.game_engine import apply_murder, assign_roles, check_terminal


def test_assign_roles_counts():
    rng = random.Random(42)
    roles, traitors = assign_roles(9, 2, rng)
    assert len(roles) == 9
    assert len(traitors) == 2
    assert sum(1 for r in roles.values() if r.value == "traitor") == 2


def test_tie_break_deterministic_murder():
    rng1 = random.Random(1)
    rng2 = random.Random(1)
    alive = {1, 2, 3, 4}
    traitors = {1, 2}
    votes = {1: 3, 2: 4}
    result1 = apply_murder(alive, traitors, votes, rng1)
    result2 = apply_murder(alive, traitors, votes, rng2)
    assert result1 == result2


def test_terminal_conditions():
    assert check_terminal({1, 2}, set()) == "faithful"
    assert check_terminal({1, 2}, {1}) == "traitors"
    assert check_terminal({1, 2, 3}, {1}) is None
