from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from traitors_ai.analysis import (  # noqa: E402
    analyse_experiment_1,
    compute_agent_role_summary,
    compute_banishment_outcomes,
    compute_overall_metrics,
    compute_round_summary,
    compute_suspicion_gap_over_time,
    compute_traitor_vote_rate_by_round,
    compute_voting_accuracy_by_round,
    compute_win_rate_by_role,
    load_experiment_run,
    validate_experiment_data,
)


def _make_mock_run_dir(tmp_path: Path, include_persona: bool = True) -> Path:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": "experiment_1_baseline_behaviour",
        "run_id": "run_test",
        "seed_list": [1, 2],
        "number_of_games": 2,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    per_game = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "winner": "faithful",
                "faithful_win": 1,
                "traitor_win": 0,
                "total_rounds": 4,
                "banishment_accuracy": 0.5,
                "deception_success_rate": 0.5,
                "belief_action_alignment_top1": 0.8,
                "belief_action_alignment_top2": 0.9,
                "suspicion_gap": 0.2,
                "traitor_vote_agreement_rate": 0.4,
                "murder_vote_agreement_rate": 0.7,
                "average_public_message_length": 240,
                "structured_output_parse_failures_count": 1,
            },
            {
                "game_id": "g2",
                "winner": "traitors",
                "faithful_win": 0,
                "traitor_win": 1,
                "total_rounds": 6,
                "banishment_accuracy": 0.2,
                "deception_success_rate": 0.7,
                "belief_action_alignment_top1": 0.6,
                "belief_action_alignment_top2": 0.8,
                "suspicion_gap": 0.1,
                "traitor_vote_agreement_rate": 0.6,
                "murder_vote_agreement_rate": 0.9,
                "average_public_message_length": 280,
                "structured_output_parse_failures_count": 0,
            },
        ]
    )
    per_game.to_csv(run_dir / "per_game_metrics.csv", index=False)

    per_round = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "round": 1,
                "mean_suspicion_to_traitors": 0.4,
                "mean_suspicion_to_faithful": 0.3,
                "vote_entropy": 1.1,
                "vote_majority_size": 4,
                "deception_success_round": True,
                    "banished_role": "faithful",
            },
            {
                "game_id": "g1",
                "round": 2,
                "mean_suspicion_to_traitors": 0.5,
                "mean_suspicion_to_faithful": 0.32,
                "vote_entropy": 1.0,
                "vote_majority_size": 5,
                "deception_success_round": False,
                    "banished_role": "traitor",
            },
            {
                "game_id": "g2",
                "round": 1,
                "mean_suspicion_to_traitors": 0.42,
                "mean_suspicion_to_faithful": 0.31,
                "vote_entropy": 1.2,
                "vote_majority_size": 4,
                "deception_success_round": True,
                    "banished_role": "faithful",
            },
        ]
    )
    per_round.to_csv(run_dir / "per_round_metrics.csv", index=False)

    per_agent_rows = [
        {
            "game_id": "g1",
            "agent_id": 1,
            "role": "faithful",
            "survived_rounds": 3,
            "average_public_message_length": 200,
            "total_public_messages": 4,
            "belief_action_alignment_top1": 0.7,
            "belief_action_alignment_top2": 0.9,
            "mean_suspicion_given_to_traitors": 0.5,
            "mean_suspicion_given_to_faithful": 0.3,
            "times_accused_by_others": 2,
            "times_voted_against": 3,
            "parse_failures": 0,
            "fallbacks_used": 1,
        },
        {
            "game_id": "g1",
            "agent_id": 2,
            "role": "traitor",
            "survived_rounds": 4,
            "average_public_message_length": 250,
            "total_public_messages": 5,
            "belief_action_alignment_top1": 0.6,
            "belief_action_alignment_top2": 0.8,
            "mean_suspicion_given_to_traitors": 0.4,
            "mean_suspicion_given_to_faithful": 0.35,
            "times_accused_by_others": 3,
            "times_voted_against": 4,
            "parse_failures": 1,
            "fallbacks_used": 0,
        },
        {
            "game_id": "g2",
            "agent_id": 1,
            "role": "faithful",
            "survived_rounds": 5,
            "average_public_message_length": 230,
            "total_public_messages": 6,
            "belief_action_alignment_top1": 0.65,
            "belief_action_alignment_top2": 0.85,
            "mean_suspicion_given_to_traitors": 0.45,
            "mean_suspicion_given_to_faithful": 0.31,
            "times_accused_by_others": 1,
            "times_voted_against": 2,
            "parse_failures": 0,
            "fallbacks_used": 0,
        },
    ]
    if include_persona:
        per_agent_rows[0]["persona_name"] = "Analyst"
        per_agent_rows[1]["persona_name"] = "Strategist"
        per_agent_rows[2]["persona_name"] = "Analyst"

    per_agent = pd.DataFrame(per_agent_rows)
    per_agent.to_csv(run_dir / "per_agent_metrics.csv", index=False)

    return run_dir


def test_load_and_validate_mock_run_dir(tmp_path: Path) -> None:
    run_dir = _make_mock_run_dir(tmp_path)
    loaded, ctx = load_experiment_run(run_dir=run_dir)
    validated = validate_experiment_data(loaded, ctx)

    assert loaded.per_game is not None
    assert loaded.per_round is not None
    assert loaded.per_agent is not None
    assert validated.row_counts["per_game_metrics.csv"] == 2
    assert "per_game_metrics.csv" in validated.required_columns_present
    assert validated.required_columns_absent["per_game_metrics.csv"] == []


def test_compute_overall_and_round_summary(tmp_path: Path) -> None:
    run_dir = _make_mock_run_dir(tmp_path)
    loaded, ctx = load_experiment_run(run_dir=run_dir)
    validate_experiment_data(loaded, ctx)

    overall_df, diss_1, diss_2 = compute_overall_metrics(loaded.per_game)
    round_summary = compute_round_summary(loaded.per_round)

    assert int(overall_df.at[0, "n_games"]) == 2
    assert overall_df.at[0, "faithful_win_rate"] == pytest.approx(0.5)
    assert overall_df.at[0, "mean_rounds"] == pytest.approx(5.0)
    assert "mean_banishment_accuracy" in diss_1.columns
    assert "mean_suspicion_gap" in diss_2.columns
    assert set(["round", "mean_suspicion_to_traitors", "average_vote_entropy"]).issubset(round_summary.columns)


def test_agent_summary_handles_missing_persona(tmp_path: Path) -> None:
    run_dir = _make_mock_run_dir(tmp_path, include_persona=False)
    loaded, ctx = load_experiment_run(run_dir=run_dir)
    validate_experiment_data(loaded, ctx)

    role_summary, persona_summary = compute_agent_role_summary(loaded.per_agent, ctx)

    assert not role_summary.empty
    assert persona_summary.empty
    assert any("persona" in w.lower() for w in ctx.warnings)


def test_analyse_experiment_writes_outputs_and_figures(tmp_path: Path) -> None:
    run_dir = _make_mock_run_dir(tmp_path)
    result = analyse_experiment_1(run_dir=run_dir, dpi=120)

    analysis_dir = Path(result["analysis_dir"])
    assert (analysis_dir / "tables" / "overall_metrics.csv").exists()
    assert (analysis_dir / "tables" / "round_summary.csv").exists()
    assert (analysis_dir / "tables" / "agent_role_summary.csv").exists()
    assert (analysis_dir / "text" / "results_summary.md").exists()
    assert (analysis_dir / "diagnostics" / "validation_report.json").exists()

    # Primary research figures
    assert (analysis_dir / "figures" / "fig_1_win_rate_by_role.png").exists()
    # fig_3 requires events.jsonl which is absent in mock; it is skipped gracefully
    # Primary figure data tables
    assert (analysis_dir / "tables" / "fig_1_win_rate_by_role.csv").exists()
    assert (analysis_dir / "tables" / "fig_3_voting_accuracy_by_round.csv").exists()


def test_compute_suspicion_gap_over_time() -> None:
    per_round = pd.DataFrame([
        {"game_id": "g1", "round": 1, "mean_suspicion_to_traitors": 0.6, "mean_suspicion_to_faithful": 0.4},
        {"game_id": "g1", "round": 2, "mean_suspicion_to_traitors": 0.7, "mean_suspicion_to_faithful": 0.4},
        {"game_id": "g2", "round": 1, "mean_suspicion_to_traitors": 0.5, "mean_suspicion_to_faithful": 0.45},
        {"game_id": "g2", "round": 2, "mean_suspicion_to_traitors": 0.65, "mean_suspicion_to_faithful": 0.42},
    ])
    result = compute_suspicion_gap_over_time(per_round)

    assert list(result["round"]) == [1, 2]
    # Round 1 gap: mean of (0.6-0.4) and (0.5-0.45) = mean(0.2, 0.05) = 0.125
    assert result.loc[result["round"] == 1, "suspicion_gap"].iloc[0] == pytest.approx(0.125)
    assert (result["suspicion_gap"] > 0).all()
    assert "se_gap" in result.columns
    assert "n_games" in result.columns


def test_compute_traitor_vote_rate_by_round() -> None:
    game_events_data = [
        (
            [
                {"action_type": "round_start", "round": 1, "payload": {"round": 1, "alive_count": 5, "traitors_alive": 1}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 2}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 2}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 3}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 3}},
            ],
            {"game_id": "g1", "roles": {"1": "faithful", "2": "traitor", "3": "faithful", "4": "faithful", "5": "faithful"}},
        )
    ]
    result = compute_traitor_vote_rate_by_round(game_events_data)

    assert len(result) == 1
    assert int(result.at[0, "total_votes"]) == 4
    assert int(result.at[0, "votes_for_traitors"]) == 2
    assert result.at[0, "traitor_vote_rate"] == pytest.approx(0.5)
    # random baseline: 1 traitor / (5-1) others = 0.25
    assert result.at[0, "random_baseline"] == pytest.approx(0.25)


def test_compute_banishment_outcomes() -> None:
    per_round = pd.DataFrame([
        {"game_id": "g1", "round": 1, "banished_role": "faithful"},
        {"game_id": "g1", "round": 2, "banished_role": "traitor"},
        {"game_id": "g2", "round": 1, "banished_role": "faithful"},
        {"game_id": "g2", "round": 2, "banished_role": "faithful"},
        {"game_id": "g2", "round": 3, "banished_role": None},  # no banishment
    ])
    result = compute_banishment_outcomes(per_round)

    assert set(result["role"]) == {"faithful", "traitor"}
    faithful_row = result[result["role"] == "faithful"].iloc[0]
    traitor_row = result[result["role"] == "traitor"].iloc[0]
    assert int(faithful_row["count"]) == 3
    assert int(traitor_row["count"]) == 1
    assert faithful_row["proportion"] == pytest.approx(0.75)
    assert traitor_row["proportion"] == pytest.approx(0.25)


def test_new_primary_figures_created(tmp_path: Path) -> None:
    """Fig 1 created from mock per_game data; fig 3 skipped (no events.jsonl in mock)."""
    run_dir = _make_mock_run_dir(tmp_path)
    result = analyse_experiment_1(run_dir=run_dir, dpi=72)
    figures = result["figures_created"]
    assert "fig_1_win_rate_by_role" in figures
    # fig_3 requires events.jsonl; skipped gracefully when absent
    assert "fig_3_voting_accuracy_by_round" not in figures
    assert "fig_3_voting_accuracy_by_round" in result["figures_skipped"]


def test_compute_win_rate_by_role() -> None:
    per_game = pd.DataFrame([
        {"game_id": "g1", "winner": "faithful"},
        {"game_id": "g2", "winner": "traitors"},
        {"game_id": "g3", "winner": "traitors"},
    ])
    result = compute_win_rate_by_role(per_game)
    assert set(result["role"]) == {"Faithful", "Traitors"}
    faithful_row = result[result["role"] == "Faithful"].iloc[0]
    traitor_row = result[result["role"] == "Traitors"].iloc[0]
    assert int(faithful_row["wins"]) == 1
    assert int(traitor_row["wins"]) == 2
    assert faithful_row["win_rate"] == pytest.approx(1 / 3)
    assert traitor_row["win_rate"] == pytest.approx(2 / 3)
    assert int(faithful_row["total_games"]) == 3


def test_compute_win_rate_by_role_fallback_columns() -> None:
    """Falls back to faithful_win / traitor_win boolean columns."""
    per_game = pd.DataFrame([
        {"game_id": "g1", "faithful_win": 1, "traitor_win": 0},
        {"game_id": "g2", "faithful_win": 0, "traitor_win": 1},
    ])
    result = compute_win_rate_by_role(per_game)
    assert result[result["role"] == "Faithful"].iloc[0]["win_rate"] == pytest.approx(0.5)
    assert result[result["role"] == "Traitors"].iloc[0]["win_rate"] == pytest.approx(0.5)


def test_compute_voting_accuracy_by_round() -> None:
    game_events_data = [
        (
            [
                {"action_type": "round_start", "round": 1, "payload": {"round": 1, "alive_count": 5, "traitors_alive": 1}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 2}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 2}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 3}},
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 3}},
            ],
            {"game_id": "g1", "roles": {"1": "faithful", "2": "traitor", "3": "faithful", "4": "faithful", "5": "faithful"}},
        )
    ]
    result = compute_voting_accuracy_by_round(game_events_data)
    assert len(result) == 1
    assert int(result.at[0, "total_votes"]) == 4
    assert int(result.at[0, "votes_for_traitors"]) == 2
    assert result.at[0, "voting_accuracy"] == pytest.approx(0.5)
    # random baseline: 1 traitor / (5-1) others = 0.25
    assert result.at[0, "random_baseline"] == pytest.approx(0.25)
    assert set(result.columns) >= {"round", "voting_accuracy", "contributing_games"}


def test_voting_accuracy_later_rounds_use_fewer_games() -> None:
    """Games that end early do not contribute to later round statistics."""
    # game g1: only round 1; game g2: rounds 1 and 2
    game_events_data = [
        (
            [
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 2}},
            ],
            {"game_id": "g1", "roles": {"1": "faithful", "2": "traitor"}},
        ),
        (
            [
                {"action_type": "vote", "phase": "voting", "round": 1, "payload": {"target_id": 2}},
                {"action_type": "vote", "phase": "voting", "round": 2, "payload": {"target_id": 1}},
            ],
            {"game_id": "g2", "roles": {"1": "faithful", "2": "traitor"}},
        ),
    ]
    result = compute_voting_accuracy_by_round(game_events_data)
    assert 2 in result["round"].values
    round_2 = result[result["round"] == 2].iloc[0]
    assert int(round_2["contributing_games"]) == 1  # only g2 reached round 2
