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
from typing import Any, Dict, List, Optional, Tuple

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
