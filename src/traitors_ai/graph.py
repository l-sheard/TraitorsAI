from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from .agent import TraitorsAgent
from .game_engine import apply_murder, apply_vote, check_terminal
from .logging_utils import JsonlLogger
from .schemas import GameState, PublicMessage, validate_vote_action


def _public_summary(messages: List[PublicMessage], player_names: Optional[Dict[int, str]] = None, max_chars: int = 600) -> str:
    if not messages:
        return "No public messages yet."
    tail = messages[-6:]
    def label(pid: int) -> str:
        return player_names.get(pid, f"P{pid}") if player_names else f"P{pid}"
    joined = " ".join([f"{label(m.speaker_id)}: {m.content}" for m in tail])
    return joined[-max_chars:]


def _traitor_summary(messages: List[PublicMessage], player_names: Optional[Dict[int, str]] = None, max_chars: int = 400) -> str:
    if not messages:
        return "No private traitor messages yet."
    tail = messages[-6:]
    def label(pid: int) -> str:
        return player_names.get(pid, f"P{pid}") if player_names else f"P{pid}"
    joined = " ".join([f"{label(m.speaker_id)}: {m.content}" for m in tail])
    return joined[-max_chars:]


def _top_k_suspicion(scores: Dict[int, float], self_id: int, k: int) -> List[int]:
    """Return the top-k most suspicious player ids (excluding self)."""
    filtered = {pid: s for pid, s in scores.items() if pid != self_id}
    return sorted(filtered, key=lambda x: filtered[x], reverse=True)[:k]


_ALIAS_POOL = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Isaac", "Julia",
    "Kevin", "Lila", "Mason", "Nora", "Owen", "Priya", "Quinn", "Ruby", "Sam", "Tara",
    "Uma", "Victor", "Willa", "Xander", "Yasmin", "Zane", "Ava", "Ben", "Clara", "Daniel",
    "Elena", "Felix", "Grace", "Henry", "Ivy", "Jonah", "Kira", "Leo", "Mia", "Noah",
]


def _llm_alias_map(state: GameState) -> Dict[int, str]:
    """Deterministic per-game aliasing for LLM prompts to reduce numeric ID bias."""
    all_ids = sorted(state.roles.keys())
    rng = random.Random(state.config.seed)
    pool = _ALIAS_POOL.copy()
    rng.shuffle(pool)

    aliases: Dict[int, str] = {}
    for idx, pid in enumerate(all_ids):
        if idx < len(pool):
            aliases[pid] = pool[idx]
        else:
            aliases[pid] = f"Player{idx + 1}"
    return aliases


def build_graph(agents: Dict[int, TraitorsAgent], logger: JsonlLogger):  # noqa: C901
    def discussion_node(state: GameState) -> GameState:
        # Log round_start event on first entry into discussion for this round
        alive_ids = sorted(state.alive)
        aliases = _llm_alias_map(state)
        alive_traitors = sorted(state.traitors & state.alive)
        alive_faithful = sorted(state.alive - state.traitors)
        state.round_tracking.append({
            "round": state.round_idx,
            "alive_count": len(alive_ids),
            "traitors_alive": len(alive_traitors),
            "faithful_alive": len(alive_faithful),
        })
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=state.round_idx,
            phase="round_start",
            actor_id=-1,
            action_type="round_start",
            payload={
                "round": state.round_idx,
                "alive_count": len(alive_ids),
                "traitors_alive": len(alive_traitors),
                "faithful_alive": len(alive_faithful),
                "alive_ids": alive_ids,
            },
        )
        print(f"Round {state.round_idx} - Discussion phase ({len(state.alive)} alive)")
        player_names = {pid: aliases[pid] for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript, player_names)
        for pid in alive_ids:
            agent = agents[pid]
            private_state = state.agent_states[pid]
            actor_role = state.roles[pid].value
            view = agent.build_view(
                round_idx=state.round_idx,
                alive_ids=alive_ids,
                player_names=player_names,
                public_summary=public_summary,
                private_state=private_state,
                traitor_ids=sorted(state.traitors),
                rng=state.rng,
            )
            if state.config.condition_name != "no_memory":
                belief_update, error = agent.update_beliefs(view)
                is_fallback = belief_update.notes == "fallback neutral"
                normalized = {
                    other: belief_update.scores.get(other, 0.5)
                    for other in alive_ids
                    if other != pid
                }
                private_state.suspicion_scores = normalized
                private_state.last_rationale = belief_update.notes
                if error:
                    state.llm_error_count += 1
                if is_fallback:
                    state.belief_update_fallback_count += 1
                    state.parse_failure_count += 1
                # Build sorted list for logging
                top_sorted = _top_k_suspicion(normalized, pid, len(normalized))
                logger.log_event(
                    game_id=state.game_id,
                    seed=state.config.seed,
                    condition=state.config.condition_name,
                    round_idx=state.round_idx,
                    phase="belief_update",
                    actor_id=pid,
                    action_type="belief_update",
                    payload={
                        "scores": {str(k): v for k, v in belief_update.scores.items()},
                        "notes": belief_update.notes,
                        "top_suspicious_sorted": top_sorted,
                        "error": error,
                        "is_fallback": is_fallback,
                    },
                    actor_role=actor_role,
                )
            for _ in range(state.config.discussion_turns):
                content = agent.speak(view)
                message = PublicMessage(
                    round=state.round_idx,
                    phase="discussion",
                    speaker_id=pid,
                    content=content,
                )
                state.public_transcript.append(message)
                logger.log_event(
                    game_id=state.game_id,
                    seed=state.config.seed,
                    condition=state.config.condition_name,
                    round_idx=state.round_idx,
                    phase="discussion",
                    actor_id=pid,
                    action_type="public_message",
                    payload={
                        "content": content,
                        "char_length": len(content),
                        "round": message.round,
                        "phase": message.phase,
                        "speaker_id": pid,
                    },
                    actor_role=actor_role,
                )
        state.phase = "discussion"
        return state

    def voting_node(state: GameState) -> GameState:
        alive_ids = sorted(state.alive)
        aliases = _llm_alias_map(state)
        player_names = {pid: aliases[pid] for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript, player_names)
        votes: Dict[int, int] = {}
        for pid in alive_ids:
            agent = agents[pid]
            private_state = state.agent_states[pid]
            actor_role = state.roles[pid].value
            view = agent.build_view(
                round_idx=state.round_idx,
                alive_ids=alive_ids,
                player_names=player_names,
                public_summary=public_summary,
                private_state=private_state,
                traitor_ids=sorted(state.traitors),
                rng=state.rng,
            )
            # Capture top-k suspicions before vote for belief-action alignment
            suspicion_snapshot = private_state.suspicion_scores or {}
            top2 = _top_k_suspicion(suspicion_snapshot, pid, 2)
            top1_suspicious = top2[0] if len(top2) >= 1 else None
            top2_suspicious = top2[1] if len(top2) >= 2 else None

            vote_action, error = agent.vote(view)
            is_valid = True
            is_fallback = False
            try:
                validate_vote_action(vote_action, pid, state.alive)
                target = vote_action.target_id
            except Exception:  # noqa: BLE001
                is_valid = False
                is_fallback = True
                candidates = [cid for cid in alive_ids if cid != pid]
                target = state.rng.choice(candidates)
            if error:
                state.llm_error_count += 1
            if is_fallback:
                state.vote_fallback_count += 1
                state.parse_failure_count += 1
            votes[pid] = target
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="voting",
                actor_id=pid,
                action_type="vote",
                payload={
                    "target_id": target,
                    "rationale": vote_action.rationale,
                    "error": error,
                    "is_fallback": is_fallback,
                    "top1_suspicious": top1_suspicious,
                    "top2_suspicious": top2_suspicious,
                },
                actor_role=actor_role,
            )
        state.vote_history.append({"round": state.round_idx, "votes": votes})
        state.phase = "voting"
        return state

    def banish_node(state: GameState) -> GameState:
        votes = state.vote_history[-1]["votes"] if state.vote_history else {}
        eliminated, tie_info = apply_vote(state.alive, votes, state.rng)
        if eliminated is None and tie_info.get("tied"):
            tied = sorted(tie_info["tied"])
            revote: Dict[int, int] = {}
            for pid in sorted(state.alive):
                agent = agents[pid]
                private_state = state.agent_states[pid]
                actor_role = state.roles[pid].value
                alive_sorted = sorted(state.alive)
                aliases = _llm_alias_map(state)
                player_names = {cid: aliases[cid] for cid in alive_sorted}
                view = agent.build_view(
                    round_idx=state.round_idx,
                    alive_ids=alive_sorted,
                    player_names=player_names,
                    public_summary=_public_summary(state.public_transcript, player_names),
                    private_state=private_state,
                    traitor_ids=sorted(state.traitors),
                    allowed_targets=tied,
                    rng=state.rng,
                )
                action, error = agent.vote(view)
                is_fallback = False
                target = action.target_id
                if target not in tied or target == pid:
                    is_fallback = True
                    choices = [cid for cid in tied if cid != pid]
                    target = state.rng.choice(choices)
                if error:
                    state.llm_error_count += 1
                if is_fallback:
                    state.vote_fallback_count += 1
                revote[pid] = target
                logger.log_event(
                    game_id=state.game_id,
                    seed=state.config.seed,
                    condition=state.config.condition_name,
                    round_idx=state.round_idx,
                    phase="revote",
                    actor_id=pid,
                    action_type="vote",
                    payload={
                        "target_id": target,
                        "rationale": action.rationale,
                        "error": error,
                        "is_fallback": is_fallback,
                    },
                    actor_role=actor_role,
                )
            eliminated, tie_info = apply_vote(state.alive, revote, state.rng)
            if eliminated is None:
                eliminated = state.rng.choice(tied)
                tie_info["random"] = True
        if eliminated is not None:
            state.alive.remove(eliminated)
            state.eliminated_order.append(eliminated)
            if state.round_tracking:
                state.round_tracking[-1]["banished_player"] = eliminated
            role = state.roles[eliminated].value
            print(f"   \u2696\ufe0f  Banished: P{eliminated} ({role})")
        eliminated_role = state.roles[eliminated].value if eliminated is not None else None
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=state.round_idx,
            phase="banish",
            actor_id=-1,
            action_type="banish_result",
            payload={
                "eliminated": eliminated,
                "eliminated_role": eliminated_role,
                "tie_info": tie_info,
            },
        )
        state.phase = "post_banish"
        return state

    def traitor_chat_node(state: GameState) -> GameState:
        if not state.traitors:
            state.phase = "traitor_chat"
            return state
        alive_traitors = sorted(state.traitors & state.alive)
        alive_ids = sorted(state.alive)
        aliases = _llm_alias_map(state)
        player_names = {pid: aliases[pid] for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript, player_names)
        traitor_summary = _traitor_summary(state.traitor_private_transcript, player_names)
        for pid in alive_traitors:
            agent = agents[pid]
            private_state = state.agent_states[pid]
            view = agent.build_view(
                round_idx=state.round_idx,
                alive_ids=alive_ids,
                player_names=player_names,
                public_summary=public_summary,
                private_state=private_state,
                traitor_ids=alive_traitors,
                traitor_summary=traitor_summary,
                rng=state.rng,
            )
            content = agent.traitor_chat(view)
            message = PublicMessage(
                round=state.round_idx,
                phase="traitor_chat",
                speaker_id=pid,
                content=content,
            )
            state.traitor_private_transcript.append(message)
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="traitor_chat",
                actor_id=pid,
                action_type="traitor_chat",
                payload={
                    "content": content,
                    "char_length": len(content),
                    "round": message.round,
                    "phase": message.phase,
                    "speaker_id": pid,
                },
                actor_role="traitor",
            )
        state.phase = "traitor_chat"
        return state

    def murder_node(state: GameState) -> GameState:
        if not state.traitors:
            state.phase = "post_murder"
            return state
        alive_traitors = sorted(state.traitors & state.alive)
        alive_ids = sorted(state.alive)
        aliases = _llm_alias_map(state)
        player_names = {pid: aliases[pid] for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript, player_names)
        traitor_summary = _traitor_summary(state.traitor_private_transcript, player_names)
        murder_votes: Dict[int, int] = {}
        for pid in alive_traitors:
            agent = agents[pid]
            private_state = state.agent_states[pid]
            view = agent.build_view(
                round_idx=state.round_idx,
                alive_ids=alive_ids,
                player_names=player_names,
                public_summary=public_summary,
                private_state=private_state,
                traitor_ids=alive_traitors,
                traitor_summary=traitor_summary,
                rng=state.rng,
            )
            action, error = agent.choose_murder(view)
            is_fallback = False
            target = action.target_id
            if target in state.traitors or target not in state.alive or target == pid:
                is_fallback = True
                candidates = [cid for cid in alive_ids if cid not in state.traitors]
                target = state.rng.choice(candidates)
            if error:
                state.llm_error_count += 1
            if is_fallback:
                state.murder_fallback_count += 1
                state.parse_failure_count += 1
            murder_votes[pid] = target
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="murder",
                actor_id=pid,
                action_type="murder",
                payload={
                    "target_id": target,
                    "rationale": action.rationale,
                    "error": error,
                    "is_fallback": is_fallback,
                },
                actor_role="traitor",
            )
        state.murder_vote_history.append({"round": state.round_idx, "murder_votes": {str(k): v for k, v in murder_votes.items()}})
        eliminated = apply_murder(state.alive, state.traitors, murder_votes, state.rng)
        if eliminated is not None:
            state.alive.remove(eliminated)
            state.eliminated_order.append(eliminated)
            if state.round_tracking:
                state.round_tracking[-1]["murdered_player"] = eliminated
            print(f"   \U0001f52a Murdered: P{eliminated} (faithful)")
        eliminated_role = state.roles[eliminated].value if eliminated is not None else None
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=state.round_idx,
            phase="murder",
            actor_id=-1,
            action_type="murder_result",
            payload={
                "eliminated": eliminated,
                "eliminated_role": eliminated_role,
                "murder_votes": {str(k): v for k, v in murder_votes.items()},
            },
        )
        state.phase = "post_murder"
        return state

    def terminal_check_node(state: GameState) -> GameState:
        winner = check_terminal(state.alive, state.traitors & state.alive)
        if not winner and state.round_idx >= state.config.max_rounds:
            winner = "draw"
        if winner:
            state.winner = winner
            alive_final = sorted(state.alive)
            traitors_final = sorted(state.alive & state.traitors)
            faithful_final = sorted(state.alive - state.traitors)
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="terminal",
                actor_id=-1,
                action_type="game_end",
                payload={
                    "winner": winner,
                    "total_rounds": state.round_idx,
                    "final_alive": alive_final,
                    "final_traitors_alive": traitors_final,
                    "final_faithful_alive": faithful_final,
                },
            )
        return state

    def route_after_terminal(state: GameState) -> str:
        if state.winner:
            return END
        if state.phase == "post_banish":
            return "traitor_chat"
        if state.phase == "post_murder":
            return "post_murder_update"
        return "discussion"

    def post_murder_update(state: GameState) -> GameState:
        alive_ids = sorted(state.alive)
        aliases = _llm_alias_map(state)
        player_names = {pid: aliases[pid] for pid in alive_ids}
        all_player_names = {pid: aliases[pid] for pid in sorted(state.roles.keys())}
        public_summary = _public_summary(state.public_transcript, player_names)
        round_messages = [message for message in state.public_transcript if message.round == state.round_idx]
        vote_record = state.vote_history[-1]["votes"] if state.vote_history else {}
        current_round = state.round_tracking[-1] if state.round_tracking else {}
        banished_player = current_round.get("banished_player")
        murdered_player = current_round.get("murdered_player")
        pre_summaries: Dict[int, str] = {}
        for pid in state.alive:
            pre_summaries[pid] = state.agent_states[pid].memory_summary
            agents[pid].update_memory_after_round(
                state.agent_states[pid],
                public_summary,
                round_idx=state.round_idx,
                public_messages=round_messages,
                vote_record=vote_record,
                banished_player=banished_player,
                murdered_player=murdered_player,
                roles=state.roles,
                alive_ids=sorted(state.alive),
                player_names=all_player_names,
            )
            new_summary = state.agent_states[pid].memory_summary
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="post_murder",
                actor_id=pid,
                action_type="memory_update",
                payload={
                    "previous_summary": pre_summaries[pid],
                    "new_summary": new_summary,
                    "summary_length": len(new_summary),
                },
                actor_role=state.roles[pid].value,
            )
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=state.round_idx,
            phase="round_end",
            actor_id=-1,
            action_type="round_end",
            payload={"round": state.round_idx},
        )
        state.round_idx += 1
        return state

    graph = StateGraph(GameState)
    graph.add_node("discussion", discussion_node)
    graph.add_node("voting", voting_node)
    graph.add_node("banish", banish_node)
    graph.add_node("terminal_check", terminal_check_node)
    graph.add_node("traitor_chat", traitor_chat_node)
    graph.add_node("murder", murder_node)
    graph.add_node("post_murder_update", post_murder_update)

    graph.set_entry_point("discussion")
    graph.add_edge("discussion", "voting")
    graph.add_edge("voting", "banish")
    graph.add_edge("banish", "terminal_check")
    graph.add_conditional_edges("terminal_check", route_after_terminal)
    graph.add_edge("traitor_chat", "murder")
    graph.add_edge("murder", "terminal_check")
    graph.add_edge("post_murder_update", "discussion")

    return graph.compile()
