from __future__ import annotations

"""
analysis.py – Experiment 1 metric computation.

All public functions accept:
  events      – list of event dicts loaded from a game's JSONL log
  game_summary – dict loaded from a game's game_summary.json (RichGameSummary shape
                 or the legacy GameSummary shape; only common fields are required)

Heuristics used for text-based metrics
---------------------------------------
accusation_rate
  A public message is counted as an accusation when it contains a player
  identifier (e.g. "P3") AND at least one of the following case-insensitive
  keywords: suspect, suspicious, traitor, lying, liar, untrustworthy.

defence_rate
  A public message is counted as a defence when it contains a player
  identifier AND at least one of: trust, innocent, faithful, defend,
  "agree with", clear, vouch.
"""

import math
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import typer

from .schemas import (
    AggregateSummary,
    PerAgentMetrics,
    PerRoundMetrics,
    RichGameSummary,
)

# ---------------------------------------------------------------------------
# Heuristic keyword lists (documented in module docstring)
# ---------------------------------------------------------------------------
_ACCUSATION_KEYWORDS = re.compile(
    r"\b(suspect|suspicious|traitor|lying|liar|untrustworthy)\b", re.IGNORECASE
)
_DEFENCE_KEYWORDS = re.compile(
    r"\b(trust|innocent|faithful|defend|clear|vouch)\b|agree with", re.IGNORECASE
)
_PLAYER_PATTERN = re.compile(r"\bP\d+\b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _vote_entropy(votes: Dict[str, int]) -> Optional[float]:
    """Shannon entropy of vote target distribution."""
    if not votes:
        return None
    target_counts: Dict[int, int] = {}
    for t in votes.values():
        target_counts[t] = target_counts.get(t, 0) + 1
    total = len(votes)
    entropy = 0.0
    for count in target_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _top_k_by_score(scores: Dict[str, float], k: int) -> List[int]:
    """Return the ids of the top-k highest-scored players (string keys -> int ids)."""
    parsed = {int(pid): float(score) for pid, score in scores.items()}
    sorted_ids = sorted(parsed, key=lambda x: parsed[x], reverse=True)
    return sorted_ids[:k]


# ---------------------------------------------------------------------------
# Core per-game metric computation
# ---------------------------------------------------------------------------

def compute_game_metrics(
    events: List[Dict[str, Any]],
    game_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all per-game Experiment 1 metrics from events + game_summary.

    Returns a flat dict suitable for CSV serialisation.
    """
    roles: Dict[int, str] = {int(k): v for k, v in game_summary.get("roles", {}).items()}
    traitors: List[int] = [pid for pid, r in roles.items() if r == "traitor"]
    faithful: List[int] = [pid for pid, r in roles.items() if r == "faithful"]
    traitor_set = set(traitors)
    faithful_set = set(faithful)
    eliminated_order: List[int] = game_summary.get("eliminated_order", [])
    winner = game_summary.get("winner")
    total_rounds = game_summary.get("total_rounds") or game_summary.get("rounds", 0)
    game_id = game_summary["game_id"]
    seed = game_summary["seed"]
    n_players = len(roles)

    # --- Primary metrics ---
    final_alive = set(roles.keys()) - set(eliminated_order)
    final_traitors_alive = sorted(final_alive & traitor_set)
    final_faithful_alive = sorted(final_alive & faithful_set)

    # --- Banish events ---
    banish_events = [e for e in events if e.get("action_type") == "banish_result"]
    total_banishments = sum(
        1 for e in banish_events
        if e["payload"].get("eliminated") is not None
    )
    traitors_banished = sum(
        1 for e in banish_events
        if e["payload"].get("eliminated") is not None
        and roles.get(e["payload"]["eliminated"]) == "traitor"
    )
    faithful_banished = sum(
        1 for e in banish_events
        if e["payload"].get("eliminated") is not None
        and roles.get(e["payload"]["eliminated"]) == "faithful"
    )
    banishment_accuracy = traitors_banished / total_banishments if total_banishments else None

    first_traitor_banish_round: Optional[int] = None
    for e in banish_events:
        elim = e["payload"].get("eliminated")
        if elim is not None and roles.get(elim) == "traitor":
            first_traitor_banish_round = e["round"]
            break

    # --- Deception success rate ---
    # Per round: at least one traitor alive after discussion + banished player is faithful
    round_starts = {e["payload"]["round"]: e["payload"] for e in events if e.get("action_type") == "round_start"}
    deception_rounds_eligible = 0
    deception_rounds_success = 0
    for e in banish_events:
        elim = e["payload"].get("eliminated")
        r = e["round"]
        rs_payload = round_starts.get(r, {})
        traitors_alive_this_round = rs_payload.get("traitors_alive", None)
        # Fallback: reconstruct approximate traitors alive from eliminated_order up to this round
        if traitors_alive_this_round is None:
            elim_before = [eid for eid in eliminated_order if eid != elim]
            # We approximate using the order – count traitors eliminated in rounds < r
            # But this is imprecise; prefer the round_start event
            banish_rounds_so_far = [
                be["round"] for be in banish_events if be["round"] < r and be["payload"].get("eliminated") is not None
            ]
            murder_events_so_far = [
                me for me in events
                if me.get("action_type") == "murder_result" and me["round"] < r
                and me["payload"].get("eliminated") is not None
            ]
            elims_before = (
                [be["payload"]["eliminated"] for be in banish_events if be["round"] < r and be["payload"].get("eliminated")]
                + [me["payload"]["eliminated"] for me in murder_events_so_far if me["payload"].get("eliminated")]
            )
            traitors_alive_this_round = len([t for t in traitors if t not in elims_before])
        if traitors_alive_this_round and traitors_alive_this_round > 0 and elim is not None:
            deception_rounds_eligible += 1
            if roles.get(elim) == "faithful":
                deception_rounds_success += 1
    deception_success_rate = (
        deception_rounds_success / deception_rounds_eligible
        if deception_rounds_eligible else None
    )

    # --- Belief-action alignment ---
    # For each vote event, check if the voted target is within top-k suspicion
    # We match each voter's belief_update in the same round with their vote.
    belief_events_by_agent_round: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for e in events:
        if e.get("action_type") == "belief_update":
            belief_events_by_agent_round[(e["actor_id"], e["round"])] = e

    vote_events = [e for e in events if e.get("action_type") == "vote" and e.get("phase") == "voting"]
    alignment_top1: List[float] = []
    alignment_top2: List[float] = []
    for ve in vote_events:
        voted_target = ve["payload"].get("target_id")
        if voted_target is None:
            continue
        # Try payload-embedded fields first (set by graph.py)
        top1_from_payload = ve["payload"].get("top1_suspicious")
        top2_from_payload = ve["payload"].get("top2_suspicious")
        if top1_from_payload is not None:
            alignment_top1.append(1.0 if voted_target == top1_from_payload else 0.0)
            alignment_top2.append(
                1.0 if voted_target in (top1_from_payload, top2_from_payload) else 0.0
            )
        else:
            # Reconstruct from the belief_update event for this agent in this round
            be = belief_events_by_agent_round.get((ve["actor_id"], ve["round"]))
            if be:
                scores = be["payload"].get("scores", {})
                if scores:
                    # Exclude self
                    top2 = _top_k_by_score(
                        {k: v for k, v in scores.items() if int(k) != ve["actor_id"]}, 2
                    )
                    if top2:
                        alignment_top1.append(1.0 if voted_target == top2[0] else 0.0)
                        alignment_top2.append(1.0 if voted_target in top2[:2] else 0.0)

    belief_action_alignment_top1 = _safe_mean(alignment_top1)
    belief_action_alignment_top2 = _safe_mean(alignment_top2)

    # --- Suspicion gap ---
    # Mean suspicion faithful agents assign to traitors minus that assigned to faithful
    suspicion_to_traitors: List[float] = []
    suspicion_to_faithful: List[float] = []
    for e in events:
        if e.get("action_type") != "belief_update":
            continue
        actor = e["actor_id"]
        if roles.get(actor) != "faithful":
            continue
        scores = e["payload"].get("scores", {})
        for pid_str, score in scores.items():
            pid = int(pid_str)
            if roles.get(pid) == "traitor":
                suspicion_to_traitors.append(float(score))
            elif roles.get(pid) == "faithful":
                suspicion_to_faithful.append(float(score))
    mean_susp_traitors = _safe_mean(suspicion_to_traitors)
    mean_susp_faithful = _safe_mean(suspicion_to_faithful)
    suspicion_gap = (
        mean_susp_traitors - mean_susp_faithful
        if mean_susp_traitors is not None and mean_susp_faithful is not None
        else None
    )

    # --- Communication metrics ---
    public_msgs = [e for e in events if e.get("action_type") == "public_message"]
    traitor_chat_msgs = [e for e in events if e.get("action_type") == "traitor_chat"]
    public_message_count_total = len(public_msgs)
    traitor_private_message_count_total = len(traitor_chat_msgs)
    msg_lengths = [len(e["payload"].get("content", "")) for e in public_msgs]
    average_public_message_length = _safe_mean([float(l) for l in msg_lengths])

    accusation_count = 0
    defence_count = 0
    for e in public_msgs:
        content = e["payload"].get("content", "")
        speaker = e["actor_id"]
        # Check if message mentions another player
        mentioned = [int(m[1:]) for m in _PLAYER_PATTERN.findall(content) if int(m[1:]) != speaker]
        if mentioned:
            if _ACCUSATION_KEYWORDS.search(content):
                accusation_count += 1
            if _DEFENCE_KEYWORDS.search(content):
                defence_count += 1
    accusation_rate = accusation_count / public_message_count_total if public_message_count_total else None
    defence_rate = defence_count / public_message_count_total if public_message_count_total else None

    # --- Traitor vote agreement rate ---
    rounds_with_traitor_vote_agreement: List[float] = []
    vote_rounds: Dict[int, Dict[int, int]] = {}
    for e in vote_events:
        r = e["round"]
        if r not in vote_rounds:
            vote_rounds[r] = {}
        vote_rounds[r][e["actor_id"]] = e["payload"]["target_id"]
    for r, round_votes in vote_rounds.items():
        traitor_votes_this_round = {pid: t for pid, t in round_votes.items() if pid in traitor_set}
        if len(traitor_votes_this_round) >= 2:
            targets = set(traitor_votes_this_round.values())
            rounds_with_traitor_vote_agreement.append(1.0 if len(targets) == 1 else 0.0)
    traitor_vote_agreement_rate = _safe_mean(rounds_with_traitor_vote_agreement)

    # --- Murder vote agreement rate ---
    murder_vote_events = [e for e in events if e.get("action_type") == "murder" and e.get("phase") == "murder"]
    murder_rounds: Dict[int, Dict[int, int]] = {}
    for e in murder_vote_events:
        r = e["round"]
        if r not in murder_rounds:
            murder_rounds[r] = {}
        murder_rounds[r][e["actor_id"]] = e["payload"]["target_id"]
    murder_agreement_scores: List[float] = []
    for r, mvotes in murder_rounds.items():
        if len(mvotes) >= 2:
            targets = set(mvotes.values())
            murder_agreement_scores.append(1.0 if len(targets) == 1 else 0.0)
    murder_vote_agreement_rate = _safe_mean(murder_agreement_scores)

    # --- Failure metrics ---
    # Count from event log
    parse_failure_events = [e for e in events if e.get("action_type") == "parsing_error"]
    fallback_events = [e for e in events if e.get("action_type") == "fallback_used"]
    vote_fallback_events = [e for e in events if e.get("action_type") == "vote" and e["payload"].get("is_fallback")]
    murder_fallback_events = [e for e in events if e.get("action_type") == "murder" and e["payload"].get("is_fallback")]
    belief_fallback_events = [e for e in events if e.get("action_type") == "belief_update" and e["payload"].get("is_fallback")]
    # Also include state-tracked counters from game_summary
    gs_parse_fail = game_summary.get("structured_output_parse_failures_count", len(parse_failure_events))
    gs_vote_fall = game_summary.get("vote_fallback_count", len(vote_fallback_events))
    gs_murder_fall = game_summary.get("murder_fallback_count", len(murder_fallback_events))
    gs_belief_fall = game_summary.get("belief_update_fallback_count", len(belief_fallback_events))
    gs_llm_err = game_summary.get("total_llm_errors_count", 0)
    gs_retry = game_summary.get("retries_used_count", 0)

    return {
        "game_id": game_id,
        "seed": seed,
        "condition": game_summary.get("condition", ""),
        "winner": winner,
        "faithful_win": winner == "faithful",
        "traitor_win": winner == "traitors",
        "total_rounds": total_rounds,
        "n_players": n_players,
        "n_traitors": len(traitors),
        "banishment_accuracy": banishment_accuracy,
        "first_traitor_banish_round": first_traitor_banish_round,
        "total_traitors_banished": traitors_banished,
        "total_faithful_banished": faithful_banished,
        "deception_success_rate": deception_success_rate,
        "belief_action_alignment_top1": belief_action_alignment_top1,
        "belief_action_alignment_top2": belief_action_alignment_top2,
        "suspicion_gap": suspicion_gap,
        "mean_traitor_suspicion_from_faithful": mean_susp_traitors,
        "mean_faithful_suspicion_from_faithful": mean_susp_faithful,
        "average_public_message_length": average_public_message_length,
        "accusation_rate": accusation_rate,
        "defence_rate": defence_rate,
        "public_message_count_total": public_message_count_total,
        "traitor_private_message_count_total": traitor_private_message_count_total,
        "traitor_vote_agreement_rate": traitor_vote_agreement_rate,
        "murder_vote_agreement_rate": murder_vote_agreement_rate,
        "structured_output_parse_failures_count": gs_parse_fail,
        "vote_fallback_count": gs_vote_fall,
        "murder_fallback_count": gs_murder_fall,
        "belief_update_fallback_count": gs_belief_fall,
        "total_llm_errors_count": gs_llm_err,
        "retries_used_count": gs_retry,
    }


# ---------------------------------------------------------------------------
# Per-round metrics
# ---------------------------------------------------------------------------

def compute_round_metrics(
    events: List[Dict[str, Any]],
    game_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute one row per game-round for per_round_metrics.csv."""
    roles: Dict[int, str] = {int(k): v for k, v in game_summary.get("roles", {}).items()}
    traitor_set = {pid for pid, r in roles.items() if r == "traitor"}
    game_id = game_summary["game_id"]
    seed = game_summary["seed"]

    # Index events by round
    round_starts = {e["payload"]["round"]: e["payload"] for e in events if e.get("action_type") == "round_start"}
    banish_by_round = {
        e["round"]: e["payload"]
        for e in events if e.get("action_type") == "banish_result"
    }
    murder_by_round = {
        e["round"]: e["payload"]
        for e in events if e.get("action_type") == "murder_result"
    }
    public_msgs_by_round: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "public_message":
            public_msgs_by_round.setdefault(e["round"], []).append(e)
    vote_events_by_round: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "vote" and e.get("phase") == "voting":
            vote_events_by_round.setdefault(e["round"], []).append(e)
    belief_events_by_round: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "belief_update":
            belief_events_by_round.setdefault(e["round"], []).append(e)

    all_rounds = sorted(
        set(list(round_starts.keys()) + list(banish_by_round.keys()) + list(vote_events_by_round.keys()))
    )
    rows = []
    for r in all_rounds:
        rs = round_starts.get(r, {})
        alive_start = rs.get("alive_count", None)
        traitors_start = rs.get("traitors_alive", None)
        faithful_start = rs.get("faithful_alive", None)

        # If round_start event is missing, fall back to None
        banish_info = banish_by_round.get(r, {})
        banished_player = banish_info.get("eliminated")
        banished_role = roles.get(banished_player) if banished_player is not None else None

        murder_info = murder_by_round.get(r, {})
        murdered_player = murder_info.get("eliminated")
        murdered_role = roles.get(murdered_player) if murdered_player is not None else None

        # Suspicion aggregates from faithful belief updates this round
        be_this_round = belief_events_by_round.get(r, [])
        susp_to_traitors: List[float] = []
        susp_to_faithful: List[float] = []
        for be in be_this_round:
            if roles.get(be["actor_id"]) != "faithful":
                continue
            scores = be["payload"].get("scores", {})
            for pid_str, score in scores.items():
                pid = int(pid_str)
                if roles.get(pid) == "traitor":
                    susp_to_traitors.append(float(score))
                else:
                    susp_to_faithful.append(float(score))

        # Vote stats
        votes_this_round = vote_events_by_round.get(r, [])
        vote_targets = {ve["actor_id"]: ve["payload"]["target_id"] for ve in votes_this_round}
        vote_ent = _vote_entropy(vote_targets) if vote_targets else None
        target_counts: Dict[int, int] = {}
        for t in vote_targets.values():
            target_counts[t] = target_counts.get(t, 0) + 1
        majority_size = max(target_counts.values()) if target_counts else None

        traitor_votes = {pid: t for pid, t in vote_targets.items() if pid in traitor_set}
        traitor_vote_agreement = (
            len(set(traitor_votes.values())) == 1
            if len(traitor_votes) >= 2
            else None
        )

        # Public messages
        pub_msgs_this_round = public_msgs_by_round.get(r, [])
        pub_msg_lengths = [len(e["payload"].get("content", "")) for e in pub_msgs_this_round]
        avg_msg_len = _safe_mean([float(l) for l in pub_msg_lengths])

        # Deception success: at least one traitor alive AND banished is faithful
        traitors_alive_start_val = traitors_start if traitors_start is not None else 0
        deception_success_round: Optional[bool] = None
        if banished_player is not None and traitors_alive_start_val > 0:
            deception_success_round = banished_role == "faithful"

        rows.append({
            "game_id": game_id,
            "seed": seed,
            "round": r,
            "alive_count_start": alive_start,
            "traitors_alive_start": traitors_start,
            "faithful_alive_start": faithful_start,
            "banished_player": banished_player,
            "banished_role": banished_role,
            "murdered_player": murdered_player,
            "murdered_role": murdered_role,
            "mean_suspicion_to_traitors": _safe_mean(susp_to_traitors),
            "mean_suspicion_to_faithful": _safe_mean(susp_to_faithful),
            "vote_entropy": vote_ent,
            "vote_majority_size": majority_size,
            "traitor_vote_agreement": traitor_vote_agreement,
            "public_message_count": len(pub_msgs_this_round),
            "average_message_length": avg_msg_len,
            "deception_success_round": deception_success_round,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-agent metrics
# ---------------------------------------------------------------------------

def compute_agent_metrics(
    events: List[Dict[str, Any]],
    game_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute one row per agent per game for per_agent_metrics.csv."""
    roles: Dict[int, str] = {int(k): v for k, v in game_summary.get("roles", {}).items()}
    personas: Dict[int, Any] = {int(k): v for k, v in game_summary.get("personas", {}).items()}
    traitor_set = {pid for pid, r in roles.items() if r == "traitor"}
    total_rounds = game_summary.get("total_rounds") or game_summary.get("rounds", 0)
    game_id = game_summary["game_id"]
    seed = game_summary["seed"]
    eliminated_order: List[int] = game_summary.get("eliminated_order", [])

    # Index belief_update events by agent
    belief_by_agent: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "belief_update":
            belief_by_agent.setdefault(e["actor_id"], []).append(e)

    # Vote events (voting phase only)
    vote_events_by_agent: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "vote" and e.get("phase") == "voting":
            vote_events_by_agent.setdefault(e["actor_id"], []).append(e)

    # Public messages by agent
    pub_msgs_by_agent: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "public_message":
            pub_msgs_by_agent.setdefault(e["actor_id"], []).append(e)

    # Traitor chat by agent
    traitor_chat_by_agent: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "traitor_chat":
            traitor_chat_by_agent.setdefault(e["actor_id"], []).append(e)

    # Murder votes by agent
    murder_votes_by_agent: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("action_type") == "murder" and e.get("phase") == "murder":
            murder_votes_by_agent.setdefault(e["actor_id"], []).append(e)

    # Vote targets per round per agent (for vote_consistency – not included in output but easy to add)
    # Times voted against
    times_voted_against: Dict[int, int] = {}
    for e in events:
        if e.get("action_type") == "vote":
            t = e["payload"].get("target_id")
            if t is not None:
                times_voted_against[t] = times_voted_against.get(t, 0) + 1

    # Build belief_update event index for alignment
    belief_by_agent_round: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for agent_id, bes in belief_by_agent.items():
        for be in bes:
            belief_by_agent_round[(agent_id, be["round"])] = be

    rows = []
    for agent_id, role in roles.items():
        persona = personas.get(agent_id, {})
        persona_name = persona.get("name", "") if isinstance(persona, dict) else ""

        # Elimination round
        if agent_id in eliminated_order:
            elim_idx = eliminated_order.index(agent_id)
            # Determine which round: scan banish/murder events for when this agent was eliminated
            elim_round: Optional[int] = None
            for e in events:
                if e.get("action_type") in ("banish_result", "murder_result"):
                    if e["payload"].get("eliminated") == agent_id:
                        elim_round = e["round"]
                        break
            survived_rounds = elim_round if elim_round is not None else total_rounds
        else:
            elim_round = None
            survived_rounds = total_rounds

        # Public messages
        pmsg = pub_msgs_by_agent.get(agent_id, [])
        p_lengths = [len(e["payload"].get("content", "")) for e in pmsg]
        avg_pub_len = _safe_mean([float(l) for l in p_lengths])

        # Votes
        agent_votes = vote_events_by_agent.get(agent_id, [])
        votes_for_traitors = sum(1 for ve in agent_votes if roles.get(ve["payload"]["target_id"]) == "traitor")
        votes_for_faithful = sum(1 for ve in agent_votes if roles.get(ve["payload"]["target_id"]) == "faithful")

        # Belief-action alignment
        a_top1: List[float] = []
        a_top2: List[float] = []
        for ve in agent_votes:
            voted_target = ve["payload"].get("target_id")
            if voted_target is None:
                continue
            top1_p = ve["payload"].get("top1_suspicious")
            top2_p = ve["payload"].get("top2_suspicious")
            if top1_p is not None:
                a_top1.append(1.0 if voted_target == top1_p else 0.0)
                a_top2.append(1.0 if voted_target in (top1_p, top2_p) else 0.0)
            else:
                be = belief_by_agent_round.get((agent_id, ve["round"]))
                if be:
                    scores = be["payload"].get("scores", {})
                    top2 = _top_k_by_score(
                        {k: v for k, v in scores.items() if int(k) != agent_id}, 2
                    )
                    if top2:
                        a_top1.append(1.0 if voted_target == top2[0] else 0.0)
                        a_top2.append(1.0 if voted_target in top2[:2] else 0.0)

        # Suspicion given to traitors vs faithful
        susp_given_to_traitors: List[float] = []
        susp_given_to_faithful: List[float] = []
        for be in belief_by_agent.get(agent_id, []):
            scores = be["payload"].get("scores", {})
            for pid_str, score in scores.items():
                pid = int(pid_str)
                if roles.get(pid) == "traitor":
                    susp_given_to_traitors.append(float(score))
                elif roles.get(pid) == "faithful":
                    susp_given_to_faithful.append(float(score))

        # Parse failures / fallbacks
        agent_be_list = belief_by_agent.get(agent_id, [])
        agent_vote_list = vote_events_by_agent.get(agent_id, [])
        agent_murder_list = murder_votes_by_agent.get(agent_id, [])
        parse_failures = sum(
            1 for e in (agent_be_list + agent_vote_list + agent_murder_list)
            if e["payload"].get("error") is not None
        )
        fallbacks_used = sum(
            1 for e in (agent_vote_list + agent_murder_list)
            if e["payload"].get("is_fallback", False)
        ) + sum(
            1 for e in agent_be_list
            if e["payload"].get("is_fallback", False)
        )

        # Times accused by others (public messages mentioning this agent + accusation keyword)
        accused_count = 0
        pid_str_pattern = re.compile(rf"\bP{agent_id}\b")
        for e in events:
            if e.get("action_type") == "public_message" and e["actor_id"] != agent_id:
                content = e["payload"].get("content", "")
                if pid_str_pattern.search(content) and _ACCUSATION_KEYWORDS.search(content):
                    accused_count += 1

        rows.append({
            "game_id": game_id,
            "seed": seed,
            "agent_id": agent_id,
            "role": role,
            "persona_name": persona_name,
            "survived_rounds": survived_rounds,
            "eliminated_round": elim_round,
            "total_public_messages": len(pmsg),
            "average_public_message_length": avg_pub_len,
            "total_votes_cast": len(agent_votes),
            "votes_for_traitors": votes_for_traitors,
            "votes_for_faithful": votes_for_faithful,
            "belief_action_alignment_top1": _safe_mean(a_top1),
            "belief_action_alignment_top2": _safe_mean(a_top2),
            "mean_suspicion_given_to_traitors": _safe_mean(susp_given_to_traitors),
            "mean_suspicion_given_to_faithful": _safe_mean(susp_given_to_faithful),
            "times_accused_by_others": accused_count,
            "times_voted_against": times_voted_against.get(agent_id, 0),
            "traitor_private_messages_sent": len(traitor_chat_by_agent.get(agent_id, [])),
            "murder_votes_cast": len(agent_murder_list),
            "parse_failures": parse_failures,
            "fallbacks_used": fallbacks_used,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregate experiment metrics
# ---------------------------------------------------------------------------

def aggregate_experiment_metrics(
    per_game_rows: List[Dict[str, Any]],
    experiment_name: str,
    run_id: str,
) -> Dict[str, Any]:
    """Aggregate per-game metric rows into an experiment-level summary dict."""
    n = len(per_game_rows)
    if n == 0:
        return AggregateSummary(
            experiment_name=experiment_name,
            run_id=run_id,
            n_games=0,
            mean_rounds=None,
            faithful_win_rate=None,
            traitor_win_rate=None,
            mean_banishment_accuracy=None,
            mean_deception_success_rate=None,
            mean_belief_action_alignment_top1=None,
            mean_belief_action_alignment_top2=None,
            mean_suspicion_gap=None,
            mean_traitor_vote_agreement_rate=None,
            mean_murder_vote_agreement_rate=None,
            mean_parse_failures_per_game=None,
        ).model_dump(mode="json")

    def _col_mean(key: str) -> Optional[float]:
        vals = [r[key] for r in per_game_rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _col_rate(key: str) -> Optional[float]:
        vals = [r[key] for r in per_game_rows if r.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    return AggregateSummary(
        experiment_name=experiment_name,
        run_id=run_id,
        n_games=n,
        mean_rounds=_col_mean("total_rounds"),
        faithful_win_rate=_col_rate("faithful_win"),
        traitor_win_rate=_col_rate("traitor_win"),
        mean_banishment_accuracy=_col_mean("banishment_accuracy"),
        mean_deception_success_rate=_col_mean("deception_success_rate"),
        mean_belief_action_alignment_top1=_col_mean("belief_action_alignment_top1"),
        mean_belief_action_alignment_top2=_col_mean("belief_action_alignment_top2"),
        mean_suspicion_gap=_col_mean("suspicion_gap"),
        mean_traitor_vote_agreement_rate=_col_mean("traitor_vote_agreement_rate"),
        mean_murder_vote_agreement_rate=_col_mean("murder_vote_agreement_rate"),
        mean_parse_failures_per_game=_col_mean("structured_output_parse_failures_count"),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Legacy helper (preserved for backwards compat)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Experiment 1 analysis pipeline (post-run)
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.callback()
def analysis_cli() -> None:
    """Post-run analysis commands for experiment outputs."""


@dataclass
class LoadedExperimentRun:
    run_dir: Path
    manifest: Optional[Dict[str, Any]]
    per_game: Optional[Any]
    per_round: Optional[Any]
    per_agent: Optional[Any]
    summary: Optional[Any]


@dataclass
class ValidationContext:
    files_found: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    row_counts: Dict[str, int] = field(default_factory=dict)
    required_columns_present: Dict[str, List[str]] = field(default_factory=dict)
    required_columns_absent: Dict[str, List[str]] = field(default_factory=dict)
    null_counts_by_file: Dict[str, Dict[str, int]] = field(default_factory=dict)
    duplicate_row_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    skipped_outputs: Dict[str, str] = field(default_factory=dict)
    ci_method: str = "bootstrap percentile CI (n_boot=1000, alpha=0.05)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_found": self.files_found,
            "files_missing": self.files_missing,
            "row_counts": self.row_counts,
            "required_columns_present": self.required_columns_present,
            "required_columns_absent": self.required_columns_absent,
            "null_counts_by_file": self.null_counts_by_file,
            "duplicate_row_checks": self.duplicate_row_checks,
            "warnings": self.warnings,
            "skipped_outputs": self.skipped_outputs,
            "ci_method": self.ci_method,
        }


def _analysis_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pandas is required for analysis. Install with: pip install -e .[analysis]"
        ) from exc
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "matplotlib is required for analysis. Install with: pip install matplotlib"
        ) from exc
    import numpy as np  # type: ignore

    return pd, np, plt


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path, pd: Any) -> Optional[Any]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _as_numeric(series: Any, pd: Any) -> Any:
    return pd.to_numeric(series, errors="coerce")


def _bootstrap_ci(values: Sequence[float], np: Any, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return lower, upper


def _scalar_stats(values: Sequence[float], np: Any) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci_low": None,
            "ci_high": None,
        }
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci_low": None,
            "ci_high": None,
        }
    ci_low, ci_high = _bootstrap_ci(arr.tolist(), np=np)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def load_experiment_run(run_dir: Path, fail_on_missing: bool = False) -> Tuple[LoadedExperimentRun, ValidationContext]:
    pd, _, _ = _analysis_dependencies()
    ctx = ValidationContext()

    required_files = ["per_game_metrics.csv"]
    optional_files = [
        "manifest.json",
        "summary.csv",
        "summary.json",
        "per_round_metrics.csv",
        "per_agent_metrics.csv",
    ]
    for name in required_files + optional_files:
        path = run_dir / name
        if path.exists():
            ctx.files_found.append(name)
        else:
            ctx.files_missing.append(name)

    if fail_on_missing:
        missing_required = [f for f in required_files if f in ctx.files_missing]
        if missing_required:
            raise FileNotFoundError(f"Missing required files: {missing_required}")

    per_game = _read_csv_if_exists(run_dir / "per_game_metrics.csv", pd)
    per_round = _read_csv_if_exists(run_dir / "per_round_metrics.csv", pd)
    per_agent = _read_csv_if_exists(run_dir / "per_agent_metrics.csv", pd)
    summary = _read_csv_if_exists(run_dir / "summary.csv", pd)
    manifest = _read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else None

    loaded = LoadedExperimentRun(
        run_dir=run_dir,
        manifest=manifest,
        per_game=per_game,
        per_round=per_round,
        per_agent=per_agent,
        summary=summary,
    )
    return loaded, ctx


def validate_experiment_data(data: LoadedExperimentRun, ctx: ValidationContext) -> ValidationContext:
    pd, _, _ = _analysis_dependencies()

    required_cols = {
        "per_game_metrics.csv": [
            "game_id",
            "winner",
            "faithful_win",
            "traitor_win",
            "total_rounds",
            "banishment_accuracy",
            "deception_success_rate",
            "belief_action_alignment_top1",
            "belief_action_alignment_top2",
            "suspicion_gap",
            "traitor_vote_agreement_rate",
            "murder_vote_agreement_rate",
            "average_public_message_length",
            "structured_output_parse_failures_count",
        ],
        "per_round_metrics.csv": [
            "game_id",
            "round",
            "mean_suspicion_to_traitors",
            "mean_suspicion_to_faithful",
            "vote_entropy",
            "vote_majority_size",
            "deception_success_round",
        ],
        "per_agent_metrics.csv": [
            "game_id",
            "agent_id",
            "role",
            "survived_rounds",
            "average_public_message_length",
            "total_public_messages",
            "belief_action_alignment_top1",
            "belief_action_alignment_top2",
            "mean_suspicion_given_to_traitors",
            "mean_suspicion_given_to_faithful",
            "times_accused_by_others",
            "times_voted_against",
            "parse_failures",
            "fallbacks_used",
        ],
    }

    file_to_df = {
        "per_game_metrics.csv": data.per_game,
        "per_round_metrics.csv": data.per_round,
        "per_agent_metrics.csv": data.per_agent,
    }

    numeric_cols = {
        "per_game_metrics.csv": [
            "total_rounds",
            "banishment_accuracy",
            "deception_success_rate",
            "belief_action_alignment_top1",
            "belief_action_alignment_top2",
            "suspicion_gap",
            "traitor_vote_agreement_rate",
            "murder_vote_agreement_rate",
            "average_public_message_length",
            "structured_output_parse_failures_count",
        ],
        "per_round_metrics.csv": [
            "round",
            "mean_suspicion_to_traitors",
            "mean_suspicion_to_faithful",
            "vote_entropy",
            "vote_majority_size",
        ],
        "per_agent_metrics.csv": [
            "survived_rounds",
            "average_public_message_length",
            "total_public_messages",
            "belief_action_alignment_top1",
            "belief_action_alignment_top2",
            "mean_suspicion_given_to_traitors",
            "mean_suspicion_given_to_faithful",
            "times_accused_by_others",
            "times_voted_against",
            "parse_failures",
            "fallbacks_used",
        ],
    }

    for filename, df in file_to_df.items():
        if df is None:
            ctx.warnings.append(f"{filename} missing; analyses depending on this file may be skipped")
            continue

        ctx.row_counts[filename] = int(len(df))
        cols = set(df.columns)
        required = required_cols[filename]
        present = [c for c in required if c in cols]
        absent = [c for c in required if c not in cols]
        ctx.required_columns_present[filename] = present
        ctx.required_columns_absent[filename] = absent

        for col in numeric_cols[filename]:
            if col in df.columns:
                df[col] = _as_numeric(df[col], pd)

        ctx.null_counts_by_file[filename] = {
            str(col): int(df[col].isna().sum())
            for col in df.columns
        }

        if filename == "per_game_metrics.csv" and "game_id" in df.columns:
            dup_count = int(df.duplicated(subset=["game_id"]).sum())
            ctx.duplicate_row_checks[filename] = {
                "subset": ["game_id"],
                "duplicate_rows": dup_count,
            }
        elif filename == "per_round_metrics.csv" and {"game_id", "round"}.issubset(cols):
            dup_count = int(df.duplicated(subset=["game_id", "round"]).sum())
            ctx.duplicate_row_checks[filename] = {
                "subset": ["game_id", "round"],
                "duplicate_rows": dup_count,
            }
        elif filename == "per_agent_metrics.csv" and {"game_id", "agent_id"}.issubset(cols):
            dup_count = int(df.duplicated(subset=["game_id", "agent_id"]).sum())
            ctx.duplicate_row_checks[filename] = {
                "subset": ["game_id", "agent_id"],
                "duplicate_rows": dup_count,
            }

    if data.per_agent is None or "persona_name" not in data.per_agent.columns:
        ctx.warnings.append("persona_name column absent in per_agent_metrics.csv; persona analyses may be skipped")
    elif data.per_agent["persona_name"].fillna("").astype(str).str.strip().eq("").all():
        ctx.warnings.append("persona_name values are empty; persona analyses may be skipped")

    if data.per_game is None:
        raise ValueError("per_game_metrics.csv is required for Experiment 1 analysis")

    return ctx


def compute_overall_metrics(per_game: Any) -> Tuple[Any, Any, Any]:
    pd, np, _ = _analysis_dependencies()

    def _col_values(name: str) -> List[float]:
        if name not in per_game.columns:
            return []
        return [float(v) for v in per_game[name].dropna().tolist()]

    faithful_vals = _col_values("faithful_win")
    traitor_vals = _col_values("traitor_win")
    rounds_vals = _col_values("total_rounds")
    banish_vals = _col_values("banishment_accuracy")
    deception_vals = _col_values("deception_success_rate")
    top1_vals = _col_values("belief_action_alignment_top1")
    top2_vals = _col_values("belief_action_alignment_top2")
    gap_vals = _col_values("suspicion_gap")
    traitor_agree_vals = _col_values("traitor_vote_agreement_rate")
    murder_agree_vals = _col_values("murder_vote_agreement_rate")
    msg_len_vals = _col_values("average_public_message_length")
    parse_vals = _col_values("structured_output_parse_failures_count")

    stat_map = {
        "faithful_win_rate": _scalar_stats(faithful_vals, np=np),
        "traitor_win_rate": _scalar_stats(traitor_vals, np=np),
        "rounds": _scalar_stats(rounds_vals, np=np),
        "banishment_accuracy": _scalar_stats(banish_vals, np=np),
        "deception_success_rate": _scalar_stats(deception_vals, np=np),
        "belief_action_alignment_top1": _scalar_stats(top1_vals, np=np),
        "belief_action_alignment_top2": _scalar_stats(top2_vals, np=np),
        "suspicion_gap": _scalar_stats(gap_vals, np=np),
        "traitor_vote_agreement_rate": _scalar_stats(traitor_agree_vals, np=np),
        "murder_vote_agreement_rate": _scalar_stats(murder_agree_vals, np=np),
        "average_public_message_length": _scalar_stats(msg_len_vals, np=np),
        "parse_failures_per_game": _scalar_stats(parse_vals, np=np),
    }

    row = {
        "n_games": int(len(per_game)),
        "faithful_win_rate": stat_map["faithful_win_rate"]["mean"],
        "faithful_win_rate_std": stat_map["faithful_win_rate"]["std"],
        "faithful_win_rate_median": stat_map["faithful_win_rate"]["median"],
        "faithful_win_rate_min": stat_map["faithful_win_rate"]["min"],
        "faithful_win_rate_max": stat_map["faithful_win_rate"]["max"],
        "faithful_win_rate_ci_low": stat_map["faithful_win_rate"]["ci_low"],
        "faithful_win_rate_ci_high": stat_map["faithful_win_rate"]["ci_high"],
        "traitor_win_rate": stat_map["traitor_win_rate"]["mean"],
        "traitor_win_rate_std": stat_map["traitor_win_rate"]["std"],
        "traitor_win_rate_median": stat_map["traitor_win_rate"]["median"],
        "traitor_win_rate_min": stat_map["traitor_win_rate"]["min"],
        "traitor_win_rate_max": stat_map["traitor_win_rate"]["max"],
        "traitor_win_rate_ci_low": stat_map["traitor_win_rate"]["ci_low"],
        "traitor_win_rate_ci_high": stat_map["traitor_win_rate"]["ci_high"],
        "mean_rounds": stat_map["rounds"]["mean"],
        "std_rounds": stat_map["rounds"]["std"],
        "median_rounds": stat_map["rounds"]["median"],
        "min_rounds": stat_map["rounds"]["min"],
        "max_rounds": stat_map["rounds"]["max"],
        "mean_rounds_ci_low": stat_map["rounds"]["ci_low"],
        "mean_rounds_ci_high": stat_map["rounds"]["ci_high"],
        "mean_banishment_accuracy": stat_map["banishment_accuracy"]["mean"],
        "mean_banishment_accuracy_std": stat_map["banishment_accuracy"]["std"],
        "mean_banishment_accuracy_median": stat_map["banishment_accuracy"]["median"],
        "mean_banishment_accuracy_ci_low": stat_map["banishment_accuracy"]["ci_low"],
        "mean_banishment_accuracy_ci_high": stat_map["banishment_accuracy"]["ci_high"],
        "mean_deception_success_rate": stat_map["deception_success_rate"]["mean"],
        "mean_deception_success_rate_std": stat_map["deception_success_rate"]["std"],
        "mean_deception_success_rate_median": stat_map["deception_success_rate"]["median"],
        "mean_deception_success_rate_ci_low": stat_map["deception_success_rate"]["ci_low"],
        "mean_deception_success_rate_ci_high": stat_map["deception_success_rate"]["ci_high"],
        "mean_belief_action_alignment_top1": stat_map["belief_action_alignment_top1"]["mean"],
        "mean_belief_action_alignment_top1_std": stat_map["belief_action_alignment_top1"]["std"],
        "mean_belief_action_alignment_top1_median": stat_map["belief_action_alignment_top1"]["median"],
        "mean_belief_action_alignment_top1_ci_low": stat_map["belief_action_alignment_top1"]["ci_low"],
        "mean_belief_action_alignment_top1_ci_high": stat_map["belief_action_alignment_top1"]["ci_high"],
        "mean_belief_action_alignment_top2": stat_map["belief_action_alignment_top2"]["mean"],
        "mean_belief_action_alignment_top2_std": stat_map["belief_action_alignment_top2"]["std"],
        "mean_belief_action_alignment_top2_median": stat_map["belief_action_alignment_top2"]["median"],
        "mean_belief_action_alignment_top2_ci_low": stat_map["belief_action_alignment_top2"]["ci_low"],
        "mean_belief_action_alignment_top2_ci_high": stat_map["belief_action_alignment_top2"]["ci_high"],
        "mean_suspicion_gap": stat_map["suspicion_gap"]["mean"],
        "mean_suspicion_gap_std": stat_map["suspicion_gap"]["std"],
        "mean_suspicion_gap_median": stat_map["suspicion_gap"]["median"],
        "mean_suspicion_gap_ci_low": stat_map["suspicion_gap"]["ci_low"],
        "mean_suspicion_gap_ci_high": stat_map["suspicion_gap"]["ci_high"],
        "mean_traitor_vote_agreement_rate": stat_map["traitor_vote_agreement_rate"]["mean"],
        "mean_traitor_vote_agreement_rate_std": stat_map["traitor_vote_agreement_rate"]["std"],
        "mean_traitor_vote_agreement_rate_median": stat_map["traitor_vote_agreement_rate"]["median"],
        "mean_traitor_vote_agreement_rate_ci_low": stat_map["traitor_vote_agreement_rate"]["ci_low"],
        "mean_traitor_vote_agreement_rate_ci_high": stat_map["traitor_vote_agreement_rate"]["ci_high"],
        "mean_murder_vote_agreement_rate": stat_map["murder_vote_agreement_rate"]["mean"],
        "mean_murder_vote_agreement_rate_std": stat_map["murder_vote_agreement_rate"]["std"],
        "mean_murder_vote_agreement_rate_median": stat_map["murder_vote_agreement_rate"]["median"],
        "mean_murder_vote_agreement_rate_ci_low": stat_map["murder_vote_agreement_rate"]["ci_low"],
        "mean_murder_vote_agreement_rate_ci_high": stat_map["murder_vote_agreement_rate"]["ci_high"],
        "mean_average_public_message_length": stat_map["average_public_message_length"]["mean"],
        "mean_average_public_message_length_std": stat_map["average_public_message_length"]["std"],
        "mean_average_public_message_length_median": stat_map["average_public_message_length"]["median"],
        "mean_average_public_message_length_ci_low": stat_map["average_public_message_length"]["ci_low"],
        "mean_average_public_message_length_ci_high": stat_map["average_public_message_length"]["ci_high"],
        "mean_parse_failures_per_game": stat_map["parse_failures_per_game"]["mean"],
        "mean_parse_failures_per_game_std": stat_map["parse_failures_per_game"]["std"],
        "mean_parse_failures_per_game_median": stat_map["parse_failures_per_game"]["median"],
        "mean_parse_failures_per_game_ci_low": stat_map["parse_failures_per_game"]["ci_low"],
        "mean_parse_failures_per_game_ci_high": stat_map["parse_failures_per_game"]["ci_high"],
    }
    overall_df = pd.DataFrame([row])

    dissertation_table_1 = pd.DataFrame([
        {
            "faithful_win_rate": row["faithful_win_rate"],
            "traitor_win_rate": row["traitor_win_rate"],
            "mean_rounds": row["mean_rounds"],
            "mean_banishment_accuracy": row["mean_banishment_accuracy"],
            "mean_deception_success_rate": row["mean_deception_success_rate"],
        }
    ])

    dissertation_table_2 = pd.DataFrame([
        {
            "mean_belief_action_alignment_top1": row["mean_belief_action_alignment_top1"],
            "mean_belief_action_alignment_top2": row["mean_belief_action_alignment_top2"],
            "mean_suspicion_gap": row["mean_suspicion_gap"],
            "mean_traitor_vote_agreement_rate": row["mean_traitor_vote_agreement_rate"],
            "mean_murder_vote_agreement_rate": row["mean_murder_vote_agreement_rate"],
        }
    ])
    return overall_df, dissertation_table_1, dissertation_table_2


def compute_round_summary(per_round: Any) -> Any:
    pd, _, _ = _analysis_dependencies()
    if per_round is None or per_round.empty:
        return pd.DataFrame(
            columns=[
                "round",
                "mean_suspicion_to_traitors",
                "mean_suspicion_to_faithful",
                "average_vote_entropy",
                "average_majority_size",
                "deception_success_frequency",
                "games_contributing",
                "se_suspicion_to_traitors",
                "se_suspicion_to_faithful",
            ]
        )

    grouped = per_round.groupby("round", dropna=True)
    out = grouped.agg(
        mean_suspicion_to_traitors=("mean_suspicion_to_traitors", "mean"),
        mean_suspicion_to_faithful=("mean_suspicion_to_faithful", "mean"),
        average_vote_entropy=("vote_entropy", "mean"),
        average_majority_size=("vote_majority_size", "mean"),
        deception_success_frequency=("deception_success_round", "mean"),
        games_contributing=("game_id", pd.Series.nunique),
        se_suspicion_to_traitors=("mean_suspicion_to_traitors", "sem"),
        se_suspicion_to_faithful=("mean_suspicion_to_faithful", "sem"),
    ).reset_index()
    return out.sort_values("round")


def _aggregate_agent_metrics(df: Any, by: List[str]) -> Any:
    return df.groupby(by, dropna=False).agg(
        n_rows=("agent_id", "count"),
        mean_survived_rounds=("survived_rounds", "mean"),
        average_public_message_length=("average_public_message_length", "mean"),
        total_public_messages=("total_public_messages", "mean"),
        belief_action_alignment_top1=("belief_action_alignment_top1", "mean"),
        belief_action_alignment_top2=("belief_action_alignment_top2", "mean"),
        mean_suspicion_given_to_traitors=("mean_suspicion_given_to_traitors", "mean"),
        mean_suspicion_given_to_faithful=("mean_suspicion_given_to_faithful", "mean"),
        times_accused_by_others=("times_accused_by_others", "mean"),
        times_voted_against=("times_voted_against", "mean"),
        parse_failures=("parse_failures", "mean"),
        fallbacks_used=("fallbacks_used", "mean"),
    ).reset_index()


def compute_agent_role_summary(per_agent: Any, ctx: ValidationContext) -> Tuple[Any, Any]:
    pd, _, _ = _analysis_dependencies()
    if per_agent is None or per_agent.empty:
        return pd.DataFrame(), pd.DataFrame()

    role_summary = _aggregate_agent_metrics(per_agent, ["role"])

    persona_summary = pd.DataFrame()
    if "persona_name" in per_agent.columns:
        persona_non_empty = per_agent["persona_name"].fillna("").astype(str).str.strip() != ""
        if persona_non_empty.any():
            persona_rows = per_agent[persona_non_empty].copy()
            by_persona = _aggregate_agent_metrics(persona_rows, ["persona_name"])
            by_persona["grouping"] = "persona"
            by_persona["role"] = "ALL"

            by_role_persona = _aggregate_agent_metrics(persona_rows, ["role", "persona_name"])
            by_role_persona["grouping"] = "role_persona"

            persona_summary = pd.concat([by_persona, by_role_persona], ignore_index=True)
            cols = ["grouping", "role", "persona_name"] + [
                c for c in persona_summary.columns if c not in {"grouping", "role", "persona_name"}
            ]
            persona_summary = persona_summary[cols]
        else:
            ctx.warnings.append("Skipping persona summary: persona_name column exists but values are empty")
    else:
        ctx.warnings.append("Skipping persona summary: persona_name column not found")

    return role_summary, persona_summary


def _save_figure(fig: Any, figures_dir: Path, stem: str, dpi: int, export_svg: bool) -> None:
    png_path = figures_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if export_svg:
        svg_path = figures_dir / f"{stem}.svg"
        fig.savefig(svg_path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Primary research figure data helpers
# ---------------------------------------------------------------------------

def _load_game_events(run_dir: Path) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Load (events, game_summary) pairs from every games/ subdirectory under run_dir."""
    games_dir = run_dir / "games"
    if not games_dir.exists():
        return []
    result: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for game_dir in sorted(games_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        summary_path = game_dir / "game_summary.json"
        if not summary_path.exists():
            continue
        events: List[Dict[str, Any]] = []
        events_path = game_dir / "events.jsonl"
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        try:
                            events.append(json.loads(stripped))
                        except Exception:  # noqa: BLE001
                            pass
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                game_summary = json.load(f)
            result.append((events, game_summary))
        except Exception:  # noqa: BLE001
            pass
    return result


def compute_suspicion_gap_over_time(per_round: Any) -> Any:
    """Compute per-round suspicion gap (traitor suspicion minus faithful suspicion) across games.

    Assumptions:
    - per_round has one row per (game_id, round) with mean_suspicion_to_traitors
      and mean_suspicion_to_faithful drawn from agents' belief update events.
    - gap = mean_suspicion_to_traitors - mean_suspicion_to_faithful.
      Positive values mean traitors are rated more suspicious than faithful;
      an upward trend across rounds indicates improving detection over time.

    Returns a DataFrame with columns:
      round, mean_suspicion_traitors, mean_suspicion_faithful,
      suspicion_gap, se_gap, n_games
    """
    pd, _, _ = _analysis_dependencies()
    empty = pd.DataFrame(columns=[
        "round", "mean_suspicion_traitors", "mean_suspicion_faithful",
        "suspicion_gap", "se_gap", "n_games",
    ])
    if per_round is None or per_round.empty:
        return empty
    df = per_round.copy()
    for col in ("round", "mean_suspicion_to_traitors", "mean_suspicion_to_faithful"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["round", "mean_suspicion_to_traitors", "mean_suspicion_to_faithful"])
    if df.empty:
        return empty
    df["gap"] = df["mean_suspicion_to_traitors"] - df["mean_suspicion_to_faithful"]
    agg = (
        df.groupby("round", dropna=True)
        .agg(
            mean_suspicion_traitors=("mean_suspicion_to_traitors", "mean"),
            mean_suspicion_faithful=("mean_suspicion_to_faithful", "mean"),
            suspicion_gap=("gap", "mean"),
            se_gap=("gap", "sem"),
            n_games=("game_id", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("round")
    )
    return agg


def compute_traitor_vote_rate_by_round(
    game_events_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
) -> Any:
    """Compute per-round traitor vote rate from raw event logs.

    For each round in each game:
      traitor_vote_rate = votes targeting a traitor / total votes cast that round

    Random baseline = traitors_alive / (alive_count - 1): the expected fraction
    of votes hitting a traitor if each voter votes uniformly at random among
    all other alive players. Derived from round_start event payloads.

    Returns a DataFrame with columns:
      round, votes_for_traitors, total_votes,
      traitor_vote_rate, se_traitor_vote_rate, random_baseline, n_games
    """
    pd, _, _ = _analysis_dependencies()
    empty = pd.DataFrame(columns=[
        "round", "votes_for_traitors", "total_votes",
        "traitor_vote_rate", "se_traitor_vote_rate", "random_baseline", "n_games",
    ])
    if not game_events_data:
        return empty
    rows: List[Dict[str, Any]] = []
    for events, game_summary in game_events_data:
        roles = {int(k): v for k, v in game_summary.get("roles", {}).items()}
        traitor_set = {pid for pid, r in roles.items() if r == "traitor"}
        game_id = game_summary.get("game_id", "")
        round_starts = {
            e["payload"]["round"]: e["payload"]
            for e in events
            if e.get("action_type") == "round_start"
        }
        vote_by_round: Dict[int, List[Dict[str, Any]]] = {}
        for e in events:
            if e.get("action_type") == "vote" and e.get("phase") == "voting":
                vote_by_round.setdefault(e["round"], []).append(e)
        for r, votes in vote_by_round.items():
            if not votes:
                continue
            total = len(votes)
            for_traitors = sum(
                1 for v in votes if v["payload"].get("target_id") in traitor_set
            )
            rs = round_starts.get(r, {})
            alive_count = rs.get("alive_count")
            traitors_alive = rs.get("traitors_alive")
            baseline: Optional[float] = None
            if alive_count is not None and traitors_alive is not None and alive_count > 1:
                baseline = traitors_alive / (alive_count - 1)
            rows.append({
                "game_id": game_id,
                "round": r,
                "votes_for_traitors": for_traitors,
                "total_votes": total,
                "traitor_vote_rate": for_traitors / total,
                "random_baseline": baseline,
            })
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("round", dropna=True)
        .agg(
            votes_for_traitors=("votes_for_traitors", "sum"),
            total_votes=("total_votes", "sum"),
            traitor_vote_rate=("traitor_vote_rate", "mean"),
            se_traitor_vote_rate=("traitor_vote_rate", "sem"),
            random_baseline=("random_baseline", "mean"),
            n_games=("game_id", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("round")
    )
    return agg


def compute_banishment_outcomes(per_round: Any) -> Any:
    """Aggregate banishment outcomes: proportion of banishments that were traitor vs faithful.

    Uses banished_role from per_round_metrics.csv. Rounds with no banishment
    (tied vote, no elimination) are excluded.

    Returns a DataFrame with columns: role, count, proportion
    """
    pd, _, _ = _analysis_dependencies()
    empty = pd.DataFrame(columns=["role", "count", "proportion"])
    if per_round is None or per_round.empty or "banished_role" not in per_round.columns:
        return empty
    valid = per_round[per_round["banished_role"].notna()].copy()
    valid = valid[valid["banished_role"].astype(str).str.strip() != ""]
    if valid.empty:
        return empty
    counts = valid["banished_role"].value_counts().reset_index()
    counts.columns = ["role", "count"]
    counts["proportion"] = counts["count"] / counts["count"].sum()
    return counts.sort_values("role").reset_index(drop=True)


def compute_win_rate_by_role(per_game: Any) -> Any:
    """Compute overall win rate for Faithful vs Traitors across all games.

    Uses the 'winner' column if present (expected values: 'faithful' / 'traitors');
    falls back to 'faithful_win' / 'traitor_win' boolean columns.

    Returns a DataFrame with columns: role, wins, total_games, win_rate
    """
    pd, _, _ = _analysis_dependencies()
    empty = pd.DataFrame(columns=["role", "wins", "total_games", "win_rate"])
    if per_game is None or per_game.empty:
        return empty
    df = per_game.copy()
    total = len(df)
    if "winner" in df.columns:
        winner = df["winner"].astype(str).str.strip().str.lower()
        faithful_wins = int((winner == "faithful").sum())
        traitor_wins = int((winner == "traitors").sum())
    elif "faithful_win" in df.columns and "traitor_win" in df.columns:
        faithful_wins = int(pd.to_numeric(df["faithful_win"], errors="coerce").fillna(0).sum())
        traitor_wins = int(pd.to_numeric(df["traitor_win"], errors="coerce").fillna(0).sum())
    else:
        return empty
    return pd.DataFrame([
        {"role": "Faithful", "wins": faithful_wins, "total_games": total, "win_rate": faithful_wins / total if total else 0.0},
        {"role": "Traitors", "wins": traitor_wins, "total_games": total, "win_rate": traitor_wins / total if total else 0.0},
    ])


def compute_voting_accuracy_by_round(
    game_events_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
) -> Any:
    """Compute per-round voting accuracy: fraction of votes targeting actual traitors.

    Aggregates across all games. Later rounds naturally include only games that
    reached those rounds — no fabrication of missing data.

    Random baseline = traitors_alive / (alive_count - 1): the fraction expected
    under uniform random voting among all other alive players. Derived from
    round_start event payloads when available.

        Returns a DataFrame with columns:
            round, total_votes, votes_for_traitors, voting_accuracy,
            sd_voting_accuracy, random_baseline, contributing_games
    """
    pd, _, _ = _analysis_dependencies()
    empty = pd.DataFrame(columns=[
        "round", "total_votes", "votes_for_traitors", "voting_accuracy",
        "sd_voting_accuracy", "random_baseline", "contributing_games",
    ])
    if not game_events_data:
        return empty
    rows: List[Dict[str, Any]] = []
    for events, game_summary in game_events_data:
        roles = {int(k): v for k, v in game_summary.get("roles", {}).items()}
        traitor_set = {pid for pid, r in roles.items() if r == "traitor"}
        game_id = game_summary.get("game_id", "")
        round_starts = {
            e["payload"]["round"]: e["payload"]
            for e in events
            if e.get("action_type") == "round_start"
        }
        vote_by_round: Dict[int, List[Dict[str, Any]]] = {}
        for e in events:
            if e.get("action_type") == "vote" and e.get("phase") == "voting":
                vote_by_round.setdefault(e["round"], []).append(e)
        for r, votes in vote_by_round.items():
            if not votes:
                continue
            total = len(votes)
            for_traitors = sum(
                1 for v in votes if v["payload"].get("target_id") in traitor_set
            )
            rs = round_starts.get(r, {})
            alive_count = rs.get("alive_count")
            traitors_alive = rs.get("traitors_alive")
            baseline: Optional[float] = None
            if alive_count is not None and traitors_alive is not None and alive_count > 1:
                baseline = traitors_alive / (alive_count - 1)
            rows.append({
                "game_id": game_id,
                "round": r,
                "votes_for_traitors": for_traitors,
                "total_votes": total,
                "voting_accuracy": for_traitors / total,
                "random_baseline": baseline,
            })
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("round", dropna=True)
        .agg(
            total_votes=("total_votes", "sum"),
            votes_for_traitors=("votes_for_traitors", "sum"),
            voting_accuracy=("voting_accuracy", "mean"),
            sd_voting_accuracy=("voting_accuracy", "std"),
            random_baseline=("random_baseline", "mean"),
            contributing_games=("game_id", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("round")
    )
    return agg


def create_figures(
    per_game: Any,
    per_round: Any,
    per_agent: Any,
    round_summary: Any,
    overall_df: Any,
    figures_dir: Path,
    dpi: int,
    export_svg: bool,
    ctx: ValidationContext,
    win_rate_df: Optional[Any] = None,
    voting_accuracy_df: Optional[Any] = None,
) -> List[str]:
    pd, _, plt = _analysis_dependencies()
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    # Fig 1: Win Rate by Role (PRIMARY — overall game outcome)
    try:
        df = win_rate_df
        if df is None or df.empty:
            raise ValueError("win_rate_df is empty or None")
        roles = df["role"].astype(str).tolist()
        win_rates = pd.to_numeric(df["win_rate"], errors="coerce").tolist()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bars = ax.bar(roles, win_rates)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Win Rate by Role")
        ax.set_xlabel("Role")
        ax.set_ylabel("Proportion of Games Won")
        for bar, rate in zip(bars, win_rates):
            if rate is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(bar.get_height()) + 0.02,
                    f"{rate * 100:.1f}%",
                    ha="center",
                    va="bottom",
                )
        _save_figure(fig, figures_dir, "fig_1_win_rate_by_role", dpi=dpi, export_svg=export_svg)
        plt.close(fig)
        created.append("fig_1_win_rate_by_role")
    except Exception as exc:  # noqa: BLE001
        ctx.skipped_outputs["fig_1_win_rate_by_role"] = f"{exc}"

    # Fig 3: Voting Accuracy by Round (PRIMARY — detection improvement over rounds)
    try:
        df = voting_accuracy_df
        if df is None or df.empty:
            raise ValueError("voting_accuracy_df is empty or None")
        x = pd.to_numeric(df["round"], errors="coerce")
        y = pd.to_numeric(df["voting_accuracy"], errors="coerce")
        sd = pd.to_numeric(df["sd_voting_accuracy"], errors="coerce").fillna(0.0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, marker="o", linewidth=2, label="Voting accuracy")
        ax.fill_between(x, y - sd, y + sd, alpha=0.25, label="\u00b11 SD")
        ax.set_title("Voting Accuracy by Round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Proportion of Votes Targeting Traitors")
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend()
        _save_figure(fig, figures_dir, "fig_3_voting_accuracy_by_round", dpi=dpi, export_svg=export_svg)
        plt.close(fig)
        created.append("fig_3_voting_accuracy_by_round")
    except Exception as exc:  # noqa: BLE001
        ctx.skipped_outputs["fig_3_voting_accuracy_by_round"] = f"{exc}"

    return created


def _fmt_pct(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: Optional[float], digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def write_text_summaries(
    text_dir: Path,
    overall_df: Any,
    round_summary: Any,
    per_agent: Any,
) -> None:
    text_dir.mkdir(parents=True, exist_ok=True)
    row = overall_df.iloc[0].to_dict() if not overall_df.empty else {}

    faithful_win = row.get("faithful_win_rate")
    traitor_win = row.get("traitor_win_rate")
    mean_rounds = row.get("mean_rounds")
    banish_acc = row.get("mean_banishment_accuracy")
    deception = row.get("mean_deception_success_rate")
    top1 = row.get("mean_belief_action_alignment_top1")
    top2 = row.get("mean_belief_action_alignment_top2")
    susp_gap = row.get("mean_suspicion_gap")
    traitor_coord = row.get("mean_traitor_vote_agreement_rate")
    murder_coord = row.get("mean_murder_vote_agreement_rate")
    n_games = int(row.get("n_games", 0) or 0)

    results_summary = "\n".join([
        "# Experiment 1 Results Summary",
        "",
        f"- Total games analysed: {n_games}",
        f"- Faithful win rate: {_fmt_pct(faithful_win)}",
        f"- Traitor win rate: {_fmt_pct(traitor_win)}",
        f"- Mean rounds per game: {_fmt_num(mean_rounds, digits=2)}",
        f"- Mean banishment accuracy: {_fmt_num(banish_acc)}",
        f"- Mean deception success rate: {_fmt_num(deception)}",
        f"- Belief-action alignment (top-1): {_fmt_num(top1)}",
        f"- Belief-action alignment (top-2): {_fmt_num(top2)}",
        f"- Mean suspicion gap (traitor minus faithful): {_fmt_num(susp_gap)}",
        f"- Mean traitor vote agreement rate: {_fmt_num(traitor_coord)}",
        f"- Mean murder vote agreement rate: {_fmt_num(murder_coord)}",
        ])

    results_summary += "\n\n" + "\n".join([
        "## Primary Result Figures",
        "",
        "**Figure 1 — Win Rate by Role** (`fig_1_win_rate_by_role.png`)",
        "Shows the overall proportion of games won by Faithful agents vs Traitor agents.",
        "This is the headline outcome: which side wins more often under baseline conditions.",
        "",
        "**Figure 3 — Voting Accuracy by Round** (`fig_3_voting_accuracy_by_round.png`)",
        "Shows whether agents increasingly cast banishment votes against actual traitors",
        "as the game progresses, compared to a round-specific random-chance baseline.",
        "An upward trend indicates improving collective identification of traitors over time.",
    ])

    poster_summary = "\n".join([
        "# Poster Summary",
        "",
        f"- {n_games} baseline games analysed.",
        f"- Faithful win rate: {_fmt_pct(faithful_win)} (traitors: {_fmt_pct(traitor_win)}).",
        f"- Mean game length: {_fmt_num(mean_rounds, digits=2)} rounds.",
        f"- Banishment accuracy: {_fmt_num(banish_acc)}; deception success: {_fmt_num(deception)}.",
        f"- Belief-action alignment: top-1 {_fmt_num(top1)}, top-2 {_fmt_num(top2)}.",
        ])

    poster_summary += "\n\n" + "\n".join([
        "## Key Figures",
        "",
        f"- Fig 1: Win Rate by Role — Faithful: {_fmt_pct(faithful_win)}, Traitors: {_fmt_pct(traitor_win)}.",
        "  Shows which side wins more often overall.",
        "",
        "- Fig 3: Voting Accuracy by Round — whether agents increasingly vote against actual traitors.",
        "  Compared against a round-specific random-chance baseline.",
    ])

    key_lines = [
        f"Faithful agents won {_fmt_pct(faithful_win)} of games.",
        f"Traitor agents won {_fmt_pct(traitor_win)} of games.",
        f"Games lasted {_fmt_num(mean_rounds, digits=2)} rounds on average.",
        f"Mean banishment accuracy was {_fmt_num(banish_acc)}.",
        f"Deception succeeded at a mean rate of {_fmt_num(deception)}.",
        f"Votes aligned with top-1 beliefs {_fmt_num(top1)} of the time.",
        f"Votes aligned with top-2 beliefs {_fmt_num(top2)} of the time.",
        f"Mean suspicion gap was {_fmt_num(susp_gap)}.",
        f"Mean traitor vote agreement was {_fmt_num(traitor_coord)}.",
        f"Mean murder vote agreement was {_fmt_num(murder_coord)}.",
        "",
        "--- Figure interpretation ---",
        f"Figure 1 (Win Rate by Role): Faithful won {_fmt_pct(faithful_win)} of games; "
        f"Traitors won {_fmt_pct(traitor_win)}. This is the headline outcome figure.",
        "Figure 3 (Voting Accuracy by Round): shows the proportion of banishment votes "
        "targeting actual traitors each round, compared to the random-chance baseline. "
        "An upward trend indicates improving collective identification of traitors over time.",
    ]

    if round_summary is not None and not round_summary.empty:
        first = round_summary.iloc[0]
        last = round_summary.iloc[-1]
        try:
            first_t = float(first["mean_suspicion_to_traitors"])
            last_t = float(last["mean_suspicion_to_traitors"])
            direction = "increased" if last_t > first_t else "decreased"
            key_lines.append(f"Suspicion toward traitors {direction} from round {int(first['round'])} to round {int(last['round'])}.")
        except Exception:
            pass

    if per_agent is not None and not per_agent.empty and "role" in per_agent.columns:
        try:
            pd, _, _ = _analysis_dependencies()
            traitor_surv = pd.to_numeric(
                per_agent.loc[per_agent["role"] == "traitor", "survived_rounds"], errors="coerce"
            ).dropna()
            if not traitor_surv.empty:
                key_lines.append(f"Traitors survived a median of {traitor_surv.median():.2f} rounds.")
        except Exception:
            pass

    (text_dir / "results_summary.md").write_text(results_summary + "\n", encoding="utf-8")
    (text_dir / "poster_summary.md").write_text(poster_summary + "\n", encoding="utf-8")
    (text_dir / "key_findings.txt").write_text("\n".join(key_lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyse_experiment_1(
    run_dir: Path,
    export_svg: bool = False,
    include_raw_log_pass: bool = False,
    fail_on_missing: bool = False,
    dpi: int = 200,
) -> Dict[str, Any]:
    pd, _, _ = _analysis_dependencies()

    data, ctx = load_experiment_run(run_dir=run_dir, fail_on_missing=fail_on_missing)
    ctx = validate_experiment_data(data, ctx)

    analysis_dir = run_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures"
    text_dir = analysis_dir / "text"
    diagnostics_dir = analysis_dir / "diagnostics"
    for d in [tables_dir, figures_dir, text_dir, diagnostics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    overall_df, diss_1, diss_2 = compute_overall_metrics(data.per_game)
    round_summary = compute_round_summary(data.per_round)
    role_summary, persona_summary = compute_agent_role_summary(data.per_agent, ctx)

    overall_df.to_csv(tables_dir / "overall_metrics.csv", index=False)
    round_summary.to_csv(tables_dir / "round_summary.csv", index=False)
    role_summary.to_csv(tables_dir / "agent_role_summary.csv", index=False)
    persona_summary.to_csv(tables_dir / "persona_summary.csv", index=False)
    diss_1.to_csv(tables_dir / "dissertation_table_1.csv", index=False)
    diss_2.to_csv(tables_dir / "dissertation_table_2.csv", index=False)

    # Compute primary research figure data
    game_events_data = _load_game_events(run_dir)
    win_rate_df = compute_win_rate_by_role(data.per_game)
    voting_accuracy_df = compute_voting_accuracy_by_round(game_events_data)

    # Save primary figure data tables
    win_rate_df.to_csv(tables_dir / "fig_1_win_rate_by_role.csv", index=False)
    voting_accuracy_df.to_csv(tables_dir / "fig_3_voting_accuracy_by_round.csv", index=False)

    created_figures = create_figures(
        per_game=data.per_game,
        per_round=data.per_round,
        per_agent=data.per_agent,
        round_summary=round_summary,
        overall_df=overall_df,
        figures_dir=figures_dir,
        dpi=dpi,
        export_svg=export_svg,
        ctx=ctx,
        win_rate_df=win_rate_df,
        voting_accuracy_df=voting_accuracy_df,
    )

    if include_raw_log_pass:
        games_dir = run_dir / "games"
        if games_dir.exists():
            events_files = list(games_dir.glob("*/events.jsonl"))
            ctx.warnings.append(f"raw log pass enabled: discovered {len(events_files)} events.jsonl files")
        else:
            ctx.warnings.append("raw log pass enabled but games/ directory not found")

    write_text_summaries(text_dir=text_dir, overall_df=overall_df, round_summary=round_summary, per_agent=data.per_agent)

    missing_report = {
        "files_missing": ctx.files_missing,
        "null_counts_by_file": ctx.null_counts_by_file,
        "warnings": ctx.warnings,
    }
    _write_json(diagnostics_dir / "missing_data_report.json", missing_report)
    _write_json(diagnostics_dir / "validation_report.json", ctx.to_dict())

    n_games = int(overall_df.at[0, "n_games"]) if not overall_df.empty else 0
    faithful = overall_df.at[0, "faithful_win_rate"] if "faithful_win_rate" in overall_df.columns else None
    traitor = overall_df.at[0, "traitor_win_rate"] if "traitor_win_rate" in overall_df.columns else None

    result = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "n_games": n_games,
        "faithful_win_rate": faithful,
        "traitor_win_rate": traitor,
        "figures_created": created_figures,
        "figures_skipped": ctx.skipped_outputs,
        "warnings": ctx.warnings,
    }
    return result


@app.command("analyse-experiment-1")
def analyse_experiment_1_cli(
    run_dir: str = typer.Option(..., help="Path to run_<id> directory"),
    export_svg: bool = typer.Option(False, help="Also export SVG figures"),
    include_raw_log_pass: bool = typer.Option(False, help="Inspect raw events.jsonl files for diagnostics"),
    fail_on_missing: bool = typer.Option(False, help="Fail when required files are missing"),
    dpi: int = typer.Option(200, help="Figure output DPI"),
) -> None:
    result = analyse_experiment_1(
        run_dir=Path(run_dir),
        export_svg=export_svg,
        include_raw_log_pass=include_raw_log_pass,
        fail_on_missing=fail_on_missing,
        dpi=dpi,
    )
    typer.echo("Experiment 1 analysis complete")
    typer.echo(f"Run dir         : {result['run_dir']}")
    typer.echo(f"Analysis dir    : {result['analysis_dir']}")
    typer.echo(f"Games analysed  : {result['n_games']}")
    typer.echo(f"Faithful win    : {_fmt_pct(result.get('faithful_win_rate'))}")
    typer.echo(f"Traitor win     : {_fmt_pct(result.get('traitor_win_rate'))}")
    typer.echo(f"Figures created : {len(result['figures_created'])}")
    typer.echo(f"Figures skipped : {len(result['figures_skipped'])}")


if __name__ == "__main__":
    app()
