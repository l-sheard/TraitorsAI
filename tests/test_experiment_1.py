"""
Tests for Experiment 1 features.

All tests are self-contained and require no LLM API calls.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from traitors_ai.agent import TraitorsAgent
from traitors_ai.analysis import (
    aggregate_experiment_metrics,
    compute_agent_metrics,
    compute_game_metrics,
    compute_round_metrics,
)
from traitors_ai.game_engine import check_terminal
from traitors_ai.graph import build_graph
from traitors_ai.logging_utils import ExperimentOutputManager, JsonlLogger
from traitors_ai.runner import _parse_seeds
from traitors_ai.schemas import (
    AgentPrivateState,
    AggregateSummary,
    BeliefUpdate,
    GameConfig,
    GameState,
    MurderAction,
    PerAgentMetrics,
    PublicMessage,
    Role,
    RichGameSummary,
    VoteAction,
)


class _DummyLLM:
    def invoke(self, prompt: str) -> str:
        return "{}"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_game_summary(
    *,
    game_id: str = "test-1-abcd1234",
    seed: int = 1,
    condition: str = "baseline_memory",
    winner: str = "faithful",
    total_rounds: int = 5,
    roles: Dict[int, str] | None = None,
    eliminated_order: List[int] | None = None,
) -> Dict[str, Any]:
    if roles is None:
        roles = {
            1: "traitor",
            2: "traitor",
            3: "faithful",
            4: "faithful",
            5: "faithful",
            6: "faithful",
            7: "faithful",
            8: "faithful",
            9: "faithful",
        }
    if eliminated_order is None:
        eliminated_order = [1, 3, 2]
    final_alive = sorted(set(roles.keys()) - set(eliminated_order))
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": condition,
        "winner": winner,
        "total_rounds": total_rounds,
        "roles": {str(k): v for k, v in roles.items()},
        "eliminated_order": eliminated_order,
        "personas": {},
        "experiment_name": "experiment_1_baseline_behaviour",
        "model_name": "test-model",
        "temperature": 0.3,
        "config": {},
        "faithful_win": winner == "faithful",
        "traitor_win": winner == "traitors",
        "final_alive": final_alive,
        "final_traitors_alive": [p for p in final_alive if roles.get(p) == "traitor"],
        "final_faithful_alive": [p for p in final_alive if roles.get(p) == "faithful"],
    }


def _make_vote_event(
    game_id: str,
    seed: int,
    round_: int,
    actor_id: int,
    target_id: int,
    top1: int | None = None,
    top2: int | None = None,
    is_fallback: bool = False,
    error: str | None = None,
) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": "baseline_memory",
        "round": round_,
        "phase": "voting",
        "actor_id": actor_id,
        "action_type": "vote",
        "payload": {
            "target_id": target_id,
            "rationale": "test",
            "error": error,
            "is_fallback": is_fallback,
            "top1_suspicious": top1,
            "top2_suspicious": top2,
        },
        "timestamp_utc": "2026-01-01T00:00:00",
    }


def _make_belief_event(
    game_id: str, seed: int, round_: int, actor_id: int, scores: Dict[int, float]
) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": "baseline_memory",
        "round": round_,
        "phase": "belief_update",
        "actor_id": actor_id,
        "action_type": "belief_update",
        "payload": {
            "scores": {str(k): v for k, v in scores.items()},
            "notes": "test",
            "is_fallback": False,
            "error": None,
        },
        "timestamp_utc": "2026-01-01T00:00:00",
    }


def _make_banish_event(
    game_id: str, seed: int, round_: int, eliminated: int, eliminated_role: str
) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": "baseline_memory",
        "round": round_,
        "phase": "banish",
        "actor_id": eliminated,
        "action_type": "banish_result",
        "payload": {"eliminated": eliminated, "eliminated_role": eliminated_role, "tie_info": {}},
        "timestamp_utc": "2026-01-01T00:00:00",
    }


def _make_public_msg_event(
    game_id: str, seed: int, round_: int, actor_id: int, content: str
) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": "baseline_memory",
        "round": round_,
        "phase": "discussion",
        "actor_id": actor_id,
        "action_type": "public_message",
        "payload": {"content": content, "char_length": len(content)},
        "timestamp_utc": "2026-01-01T00:00:00",
    }


def _make_round_start_event(
    game_id: str, seed: int, round_: int, alive_count: int, traitors_alive: int, faithful_alive: int
) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": "baseline_memory",
        "round": round_,
        "phase": "round_start",
        "actor_id": -1,
        "action_type": "round_start",
        "payload": {
            "round": round_,
            "alive_count": alive_count,
            "traitors_alive": traitors_alive,
            "faithful_alive": faithful_alive,
            "alive_ids": list(range(1, alive_count + 1)),
        },
        "timestamp_utc": "2026-01-01T00:00:00",
    }


def _make_murder_event(
    game_id: str, seed: int, round_: int, actor_id: int, target_id: int, is_fallback: bool = False
) -> Dict[str, Any]:
    return {
        "game_id": game_id,
        "seed": seed,
        "condition": "baseline_memory",
        "round": round_,
        "phase": "murder",
        "actor_id": actor_id,
        "action_type": "murder",
        "payload": {
            "target_id": target_id,
            "rationale": "test",
            "is_fallback": is_fallback,
            "error": None,
        },
        "timestamp_utc": "2026-01-01T00:00:00",
    }


# ---------------------------------------------------------------------------
# Seed parsing
# ---------------------------------------------------------------------------


class TestParseSeeeds:
    def test_single_seed(self):
        assert _parse_seeds("42") == [42]

    def test_range(self):
        assert _parse_seeds("1..5") == [1, 2, 3, 4, 5]

    def test_single_element_range(self):
        assert _parse_seeds("7..7") == [7]

    def test_large_range_length(self):
        result = _parse_seeds("1..100")
        assert len(result) == 100
        assert result[0] == 1
        assert result[-1] == 100


# ---------------------------------------------------------------------------
# Banishment accuracy
# ---------------------------------------------------------------------------


class TestBanishmentAccuracy:
    def test_all_traitors_banished(self):
        gs = _make_game_summary(winner="faithful")
        events = [
            _make_banish_event("g1", 1, 1, 1, "traitor"),
            _make_banish_event("g1", 1, 2, 2, "traitor"),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["banishment_accuracy"] == pytest.approx(1.0)
        assert metrics["total_traitors_banished"] == 2
        assert metrics["total_faithful_banished"] == 0

    def test_no_traitors_banished(self):
        gs = _make_game_summary(winner="traitors")
        events = [
            _make_banish_event("g1", 1, 1, 3, "faithful"),
            _make_banish_event("g1", 1, 2, 4, "faithful"),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["banishment_accuracy"] == pytest.approx(0.0)

    def test_mixed_banishments(self):
        gs = _make_game_summary()
        events = [
            _make_banish_event("g1", 1, 1, 1, "traitor"),
            _make_banish_event("g1", 1, 2, 3, "faithful"),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["banishment_accuracy"] == pytest.approx(0.5)

    def test_no_banishments(self):
        gs = _make_game_summary()
        metrics = compute_game_metrics([], gs)
        assert metrics["banishment_accuracy"] is None

    def test_first_traitor_banish_round(self):
        gs = _make_game_summary()
        events = [
            _make_banish_event("g1", 1, 1, 3, "faithful"),
            _make_banish_event("g1", 1, 2, 1, "traitor"),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["first_traitor_banish_round"] == 2


# ---------------------------------------------------------------------------
# Belief-action alignment
# ---------------------------------------------------------------------------


class TestBeliefActionAlignment:
    def test_top1_alignment_perfect(self):
        gs = _make_game_summary()
        events = [
            _make_vote_event("g1", 1, 1, 3, target_id=1, top1=1, top2=2),
            _make_vote_event("g1", 1, 1, 4, target_id=2, top1=2, top2=1),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["belief_action_alignment_top1"] == pytest.approx(1.0)
        assert metrics["belief_action_alignment_top2"] == pytest.approx(1.0)

    def test_top1_alignment_zero(self):
        gs = _make_game_summary()
        events = [
            _make_vote_event("g1", 1, 1, 3, target_id=5, top1=1, top2=2),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["belief_action_alignment_top1"] == pytest.approx(0.0)
        assert metrics["belief_action_alignment_top2"] == pytest.approx(0.0)

    def test_top2_alignment_hit(self):
        gs = _make_game_summary()
        # Voted for top2 player (not top1)
        events = [
            _make_vote_event("g1", 1, 1, 3, target_id=2, top1=1, top2=2),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["belief_action_alignment_top1"] == pytest.approx(0.0)
        assert metrics["belief_action_alignment_top2"] == pytest.approx(1.0)

    def test_alignment_from_belief_update(self):
        """Alignment computed from belief_update event when top-k not in vote payload."""
        gs = _make_game_summary()
        # P3 is faithful; suspicion: P1 (traitor) = 0.9, P2 (traitor) = 0.7
        events = [
            _make_belief_event("g1", 1, 1, 3, {1: 0.9, 2: 0.7, 4: 0.3, 5: 0.2}),
            # Vote P1 → top1 → alignment=1
            _make_vote_event("g1", 1, 1, 3, target_id=1),  # no top1/top2 in payload
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["belief_action_alignment_top1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Suspicion gap
# ---------------------------------------------------------------------------


class TestSuspicionGap:
    def test_positive_gap(self):
        """Faithful agent assigns high suspicion to traitors."""
        gs = _make_game_summary()
        events = [
            # Faithful agent P3 suspicious of traitors (P1, P2) more than faithful
            _make_belief_event("g1", 1, 1, 3, {1: 0.9, 2: 0.8, 4: 0.2, 5: 0.1}),
        ]
        metrics = compute_game_metrics(events, gs)
        # mean_traitor = (0.9+0.8)/2 = 0.85; mean_faithful = (0.2+0.1)/2 = 0.15
        assert metrics["suspicion_gap"] == pytest.approx(0.85 - 0.15, abs=1e-6)

    def test_zero_gap(self):
        gs = _make_game_summary()
        events = [
            _make_belief_event("g1", 1, 1, 3, {1: 0.5, 2: 0.5, 4: 0.5, 5: 0.5}),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["suspicion_gap"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Traitor vote agreement
# ---------------------------------------------------------------------------


class TestTraitorVoteAgreement:
    def test_full_agreement(self):
        gs = _make_game_summary()
        events = [
            _make_vote_event("g1", 1, 1, 1, target_id=3, top1=3, top2=4),
            _make_vote_event("g1", 1, 1, 2, target_id=3, top1=3, top2=4),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["traitor_vote_agreement_rate"] == pytest.approx(1.0)

    def test_no_agreement(self):
        gs = _make_game_summary()
        events = [
            _make_vote_event("g1", 1, 1, 1, target_id=3),
            _make_vote_event("g1", 1, 1, 2, target_id=4),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["traitor_vote_agreement_rate"] == pytest.approx(0.0)

    def test_single_traitor_voting(self):
        """Only one traitor alive – not counted towards agreement rate."""
        gs = _make_game_summary()
        events = [
            _make_vote_event("g1", 1, 1, 1, target_id=3),  # only one traitor
        ]
        metrics = compute_game_metrics(events, gs)
        # No round has ≥2 traitors voting → rate is None
        assert metrics["traitor_vote_agreement_rate"] is None


# ---------------------------------------------------------------------------
# Communication metrics (accusation / defence heuristics)
# ---------------------------------------------------------------------------


class TestCommunicationMetrics:
    def test_accusation_rate(self):
        gs = _make_game_summary()
        events = [
            # Accusation: mentions P2 and "suspect"
            _make_public_msg_event("g1", 1, 1, 3, "I suspect P2 is a traitor."),
            # Not an accusation
            _make_public_msg_event("g1", 1, 1, 4, "Let's all vote carefully."),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["accusation_rate"] == pytest.approx(0.5)

    def test_defence_rate(self):
        gs = _make_game_summary()
        events = [
            _make_public_msg_event("g1", 1, 1, 3, "I trust P4, they seem innocent."),
            _make_public_msg_event("g1", 1, 1, 4, "P1 is suspicious."),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["defence_rate"] == pytest.approx(0.5)

    def test_message_count(self):
        gs = _make_game_summary()
        events = [
            _make_public_msg_event("g1", 1, 1, 3, "Hello"),
            _make_public_msg_event("g1", 1, 1, 4, "World"),
            _make_public_msg_event("g1", 1, 2, 3, "Test"),
        ]
        metrics = compute_game_metrics(events, gs)
        assert metrics["public_message_count_total"] == 3


# ---------------------------------------------------------------------------
# ExperimentOutputManager – directory creation
# ---------------------------------------------------------------------------


class TestExperimentOutputManager:
    def test_creates_run_directory(self, tmp_path):
        manager = ExperimentOutputManager(str(tmp_path), "testrun001")
        assert manager.run_dir.exists()
        assert manager.games_dir.exists()

    def test_game_logger_creates_subdirectory(self, tmp_path):
        manager = ExperimentOutputManager(str(tmp_path), "testrun002")
        logger = manager.game_logger("test-game-001", "gpt-4o-mini")
        assert (manager.games_dir / "test-game-001").exists()
        logger.close()

    def test_write_csv(self, tmp_path):
        manager = ExperimentOutputManager(str(tmp_path), "testrun003")
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = manager.write_csv("test.csv", rows)
        content = Path(path).read_text(encoding="utf-8")
        assert "a,b" in content
        assert "1,2" in content

    def test_write_manifest(self, tmp_path):
        manager = ExperimentOutputManager(str(tmp_path), "testrun004")
        data = {"experiment_name": "exp1", "n_games": 5}
        path = manager.write_manifest(data)
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["n_games"] == 5

    def test_make_run_id_format(self):
        run_id = ExperimentOutputManager.make_run_id()
        # Should match YYYYMMDDTHHMMSSz pattern
        assert len(run_id) == 16
        assert "T" in run_id


# ---------------------------------------------------------------------------
# Aggregate experiment metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_basic_aggregation(self):
        rows = [
            {
                "game_id": "g1",
                "seed": 1,
                "winner": "faithful",
                "faithful_win": True,
                "traitor_win": False,
                "total_rounds": 8,
                "banishment_accuracy": 0.6,
                "deception_success_rate": 0.4,
                "belief_action_alignment_top1": 0.7,
                "belief_action_alignment_top2": 0.9,
                "suspicion_gap": 0.3,
                "traitor_vote_agreement_rate": 0.5,
                "murder_vote_agreement_rate": 0.4,
                "structured_output_parse_failures_count": 0,
            },
            {
                "game_id": "g2",
                "seed": 2,
                "winner": "traitors",
                "faithful_win": False,
                "traitor_win": True,
                "total_rounds": 12,
                "banishment_accuracy": 0.2,
                "deception_success_rate": 0.8,
                "belief_action_alignment_top1": 0.3,
                "belief_action_alignment_top2": 0.5,
                "suspicion_gap": 0.1,
                "traitor_vote_agreement_rate": 0.7,
                "murder_vote_agreement_rate": 0.6,
                "structured_output_parse_failures_count": 2,
            },
        ]
        agg = aggregate_experiment_metrics(rows, "experiment_1_baseline_behaviour", "run001")
        assert agg["n_games"] == 2
        assert agg["faithful_win_rate"] == pytest.approx(0.5)
        assert agg["traitor_win_rate"] == pytest.approx(0.5)
        assert agg["mean_rounds"] == pytest.approx(10.0)
        assert agg["mean_banishment_accuracy"] == pytest.approx(0.4)
        assert agg["mean_parse_failures_per_game"] == pytest.approx(1.0)

    def test_empty_rows(self):
        agg = aggregate_experiment_metrics([], "experiment_1_baseline_behaviour", "run000")
        assert agg["n_games"] == 0
        assert agg["faithful_win_rate"] is None


# ---------------------------------------------------------------------------
# Per-round metric computation
# ---------------------------------------------------------------------------


class TestPerRoundMetrics:
    def test_round_metrics_structure(self):
        gs = _make_game_summary()
        events = [
            _make_round_start_event("g1", 1, 1, 9, 2, 7),
            _make_vote_event("g1", 1, 1, 3, target_id=1, top1=1),
            _make_vote_event("g1", 1, 1, 4, target_id=1, top1=1),
            _make_banish_event("g1", 1, 1, 1, "traitor"),
            _make_public_msg_event("g1", 1, 1, 3, "Test message"),
        ]
        rows = compute_round_metrics(events, gs)
        assert len(rows) == 1
        r = rows[0]
        assert r["round"] == 1
        assert r["banished_player"] == 1
        assert r["banished_role"] == "traitor"
        assert r["public_message_count"] == 1
        assert r["alive_count_start"] == 9
        assert r["traitors_alive_start"] == 2


# ---------------------------------------------------------------------------
# Per-agent metric computation
# ---------------------------------------------------------------------------


class TestPerAgentMetrics:
    def test_agent_metrics_structure(self):
        gs = _make_game_summary()
        events = [
            _make_vote_event(
                "g1", 1, 1, 3, target_id=1, top1=1, top2=2
            ),  # faithful P3 votes traitor P1
            _make_public_msg_event("g1", 1, 1, 3, "I suspect P1 looks untrustworthy"),
            _make_belief_event("g1", 1, 1, 3, {1: 0.8, 2: 0.6, 4: 0.2, 5: 0.1}),
        ]
        rows = compute_agent_metrics(events, gs)
        assert len(rows) == 9  # one per player
        p3 = next(r for r in rows if r["agent_id"] == 3)
        assert p3["role"] == "faithful"
        assert p3["total_votes_cast"] == 1
        assert p3["votes_for_traitors"] == 1
        assert p3["votes_for_faithful"] == 0
        assert p3["belief_action_alignment_top1"] == pytest.approx(1.0)
        assert p3["total_public_messages"] == 1


class TestAgentMemory:
    def test_structured_memory_tracks_speeches_votes_and_eliminations(self):
        config = GameConfig(seed=1, condition_name="baseline_memory")
        agent = TraitorsAgent(
            agent_id=1,
            persona={
                "name": "Tester",
                "speaking_style": ["plain"],
                "social_style": ["neutral"],
                "biases": ["none"],
                "strategy_tendencies": {},
                "catchphrases": ["test"],
            },
            role="faithful",
            llm_client=_DummyLLM(),
            config=config,
        )
        private_state = AgentPrivateState(suspicion_scores={2: 0.5, 3: 0.5, 4: 0.5})
        public_messages = [
            PublicMessage(
                round=1, phase="discussion", speaker_id=2, content="I suspect P3 is a traitor."
            ),
            PublicMessage(round=1, phase="discussion", speaker_id=3, content="I trust P4 for now."),
        ]

        agent.update_memory_after_round(
            private_state,
            "P2 accused P3. P3 defended P4.",
            round_idx=1,
            public_messages=public_messages,
            vote_record={1: 2, 2: 3, 3: 2, 4: 2},
            banished_player=4,
            murdered_player=5,
            roles={
                1: Role.faithful,
                2: Role.faithful,
                3: Role.faithful,
                4: Role.traitor,
                5: Role.faithful,
            },
            alive_ids=[1, 2, 3],
        )

        assert private_state.vote_memory[2] == [3]
        assert private_state.last_public_messages[2].startswith("I suspect P3")
        assert "R1" in private_state.memory_summary
        assert "P2:" in private_state.memory_summary
        assert "recent votes P3" in private_state.memory_summary
        assert "banished P4" in private_state.memory_summary
        assert "murdered P5" in private_state.memory_summary

    def test_no_memory_condition_clears_structured_memory(self):
        config = GameConfig(seed=1, condition_name="no_memory")
        agent = TraitorsAgent(
            agent_id=1,
            persona={
                "name": "Tester",
                "speaking_style": ["plain"],
                "social_style": ["neutral"],
                "biases": ["none"],
                "strategy_tendencies": {},
                "catchphrases": ["test"],
            },
            role="faithful",
            llm_client=_DummyLLM(),
            config=config,
        )
        private_state = AgentPrivateState(
            memory_summary="old",
            round_summaries=["R1: old"],
            player_notes={2: "old note"},
            last_public_messages={2: "old msg"},
            vote_memory={2: [3]},
            elimination_memory=["R1: banished P4"],
        )

        agent.update_memory_after_round(private_state, "ignored")

        assert private_state.memory_summary == ""
        assert private_state.round_summaries == []
        assert private_state.player_notes == {}
        assert private_state.last_public_messages == {}
        assert private_state.vote_memory == {}
        assert private_state.elimination_memory == []


# ---------------------------------------------------------------------------
# Graph elimination behavior
# ---------------------------------------------------------------------------


class _StubAgent:
    def __init__(self, agent_id: int, vote_plan: Dict[int, int], murder_plan: Dict[int, int]):
        self.id = agent_id
        self.persona = {"name": f"P{agent_id}"}
        self.vote_plan = vote_plan
        self.murder_plan = murder_plan

    def build_view(self, **kwargs):
        return {
            "round": kwargs["round_idx"],
            "alive_ids": kwargs["alive_ids"],
            "alive_names": [kwargs["player_names"][pid] for pid in kwargs["alive_ids"]],
            "public_summary": kwargs["public_summary"],
            "memory_summary": kwargs["private_state"].memory_summary,
            "top_suspicions": "none",
            "traitor_ids": kwargs["traitor_ids"],
            "traitor_summary": kwargs.get("traitor_summary", ""),
            "allowed_targets": kwargs.get("allowed_targets", []),
            "rng": kwargs["rng"],
        }

    def update_beliefs(self, view):
        scores = {pid: 0.5 for pid in view["alive_ids"] if pid != self.id}
        return BeliefUpdate(scores=scores, notes="stub"), None

    def speak(self, view):
        return f"P{self.id} round {view['round']}"

    def vote(self, view):
        return VoteAction(target_id=self.vote_plan[view["round"]], rationale="stub"), None

    def traitor_chat(self, view):
        return "traitor plan"

    def choose_murder(self, view):
        return MurderAction(target_id=self.murder_plan[view["round"]], rationale="stub"), None

    def update_memory_after_round(self, state, public_summary, **kwargs):
        state.memory_summary = public_summary[-50:]


class TestEliminationBehavior:
    def test_eliminated_players_do_not_act_in_later_rounds(self, tmp_path):
        config = GameConfig(
            seed=1,
            n_players=5,
            n_traitors=1,
            max_rounds=3,
            discussion_turns=1,
            condition_name="baseline_memory",
        )
        roles = {
            1: Role.faithful,
            2: Role.traitor,
            3: Role.faithful,
            4: Role.faithful,
            5: Role.faithful,
        }
        alive = {1, 2, 3, 4, 5}
        state = GameState(
            config=config,
            game_id="test-game",
            round_idx=1,
            alive=alive,
            roles=roles,
            traitors={2},
            public_transcript=[],
            vote_history=[],
            traitor_private_transcript=[],
            agent_states={
                pid: AgentPrivateState(
                    memory_summary="",
                    suspicion_scores={other: 0.5 for other in alive if other != pid},
                )
                for pid in alive
            },
            rng=random.Random(1),
        )
        agents = {
            1: _StubAgent(1, {1: 2}, {}),
            2: _StubAgent(2, {1: 1, 2: 3}, {1: 5}),
            3: _StubAgent(3, {1: 1, 2: 2}, {}),
            4: _StubAgent(4, {1: 1, 2: 2}, {}),
            5: _StubAgent(5, {1: 1}, {}),
        }

        logger = JsonlLogger(str(tmp_path), state.game_id)
        graph = build_graph(agents, logger)
        final_state = GameState.model_validate(graph.invoke(state))
        logger.close()

        assert (
            check_terminal(final_state.alive, final_state.traitors & final_state.alive)
            == "faithful"
        )

        events = [
            json.loads(line)
            for line in (tmp_path / "test-game.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        banish_results = [e for e in events if e["action_type"] == "banish_result"]
        murder_results = [e for e in events if e["action_type"] == "murder_result"]
        assert all(e["actor_id"] == -1 for e in banish_results)
        assert all(e["actor_id"] == -1 for e in murder_results)

        later_round_actions = [
            e
            for e in events
            if e["round"] >= 2
            and e["action_type"]
            in {"belief_update", "public_message", "vote", "traitor_chat", "murder"}
        ]
        acting_ids = {e["actor_id"] for e in later_round_actions}
        assert 1 not in acting_ids
        assert 5 not in acting_ids
