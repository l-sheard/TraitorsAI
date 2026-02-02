from __future__ import annotations

from typing import Dict, List, Tuple

from langgraph.graph import END, StateGraph

from .agent import TraitorsAgent
from .game_engine import apply_murder, apply_vote, check_terminal
from .logging_utils import JsonlLogger
from .schemas import GameState, PublicMessage, validate_vote_action


def _public_summary(messages: List[PublicMessage], max_chars: int = 600) -> str:
    if not messages:
        return "No public messages yet."
    tail = messages[-6:]
    joined = " ".join([f"P{m.speaker_id}: {m.content}" for m in tail])
    return joined[-max_chars:]


def _traitor_summary(messages: List[PublicMessage], max_chars: int = 400) -> str:
    if not messages:
        return "No private traitor messages yet."
    tail = messages[-6:]
    joined = " ".join([f"P{m.speaker_id}: {m.content}" for m in tail])
    return joined[-max_chars:]


def build_graph(agents: Dict[int, TraitorsAgent], logger: JsonlLogger):
    def discussion_node(state: GameState) -> GameState:
        print(f"Round {state.round_idx} - Discussion phase ({len(state.alive)} alive)")
        alive_ids = sorted(state.alive)
        player_names = {pid: f"P{pid}" for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript)
        for pid in alive_ids:
            agent = agents[pid]
            private_state = state.agent_states[pid]
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
                normalized = {
                    other: belief_update.scores.get(other, 0.5)
                    for other in alive_ids
                    if other != pid
                }
                private_state.suspicion_scores = normalized
                private_state.last_rationale = belief_update.notes
                logger.log_event(
                    game_id=state.game_id,
                    seed=state.config.seed,
                    condition=state.config.condition_name,
                    round_idx=state.round_idx,
                    phase="belief_update",
                    actor_id=pid,
                    action_type="belief_update",
                    payload={
                        "scores": belief_update.scores,
                        "notes": belief_update.notes,
                        "error": error,
                    },
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
                    payload=message.model_dump(),
                )
        state.phase = "discussion"
        return state

    def voting_node(state: GameState) -> GameState:
        alive_ids = sorted(state.alive)
        player_names = {pid: f"P{pid}" for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript)
        votes: Dict[int, int] = {}
        for pid in alive_ids:
            agent = agents[pid]
            private_state = state.agent_states[pid]
            view = agent.build_view(
                round_idx=state.round_idx,
                alive_ids=alive_ids,
                player_names=player_names,
                public_summary=public_summary,
                private_state=private_state,
                traitor_ids=sorted(state.traitors),
                rng=state.rng,
            )
            vote_action, error = agent.vote(view)
            try:
                validate_vote_action(vote_action, pid, state.alive)
                target = vote_action.target_id
            except Exception:  # noqa: BLE001
                candidates = [cid for cid in alive_ids if cid != pid]
                target = state.rng.choice(candidates)
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
                },
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
                player_names = {cid: f"P{cid}" for cid in sorted(state.alive)}
                view = agent.build_view(
                    round_idx=state.round_idx,
                    alive_ids=sorted(state.alive),
                    player_names=player_names,
                    public_summary=_public_summary(state.public_transcript),
                    private_state=private_state,
                    traitor_ids=sorted(state.traitors),
                    allowed_targets=tied,
                    rng=state.rng,
                )
                action, error = agent.vote(view)
                target = action.target_id
                if target not in tied or target == pid:
                    choices = [cid for cid in tied if cid != pid]
                    target = state.rng.choice(choices)
                revote[pid] = target
                logger.log_event(
                    game_id=state.game_id,
                    seed=state.config.seed,
                    condition=state.config.condition_name,
                    round_idx=state.round_idx,
                    phase="revote",
                    actor_id=pid,
                    action_type="vote",
                    payload={"target_id": target, "rationale": action.rationale, "error": error},
                )
            eliminated, tie_info = apply_vote(state.alive, revote, state.rng)
            if eliminated is None:
                eliminated = state.rng.choice(tied)
                tie_info["random"] = True
        if eliminated is not None:
            state.alive.remove(eliminated)
            state.eliminated_order.append(eliminated)
            role = state.roles[eliminated].value
            print(f"   ⚖️  Banished: P{eliminated} ({role})")
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=state.round_idx,
            phase="banish",
            actor_id=eliminated or -1,
            action_type="banish_result",
            payload={"eliminated": eliminated, "tie_info": tie_info},
        )
        state.phase = "post_banish"
        return state

    def traitor_chat_node(state: GameState) -> GameState:
        if not state.traitors:
            state.phase = "traitor_chat"
            return state
        alive_traitors = sorted(state.traitors & state.alive)
        alive_ids = sorted(state.alive)
        player_names = {pid: f"P{pid}" for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript)
        traitor_summary = _traitor_summary(state.traitor_private_transcript)
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
                payload=message.model_dump(),
            )
        state.phase = "traitor_chat"
        return state

    def murder_node(state: GameState) -> GameState:
        if not state.traitors:
            state.phase = "post_murder"
            return state
        alive_traitors = sorted(state.traitors & state.alive)
        alive_ids = sorted(state.alive)
        player_names = {pid: f"P{pid}" for pid in alive_ids}
        public_summary = _public_summary(state.public_transcript)
        traitor_summary = _traitor_summary(state.traitor_private_transcript)
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
            target = action.target_id
            if target in state.traitors or target not in state.alive or target == pid:
                candidates = [cid for cid in alive_ids if cid not in state.traitors]
                target = state.rng.choice(candidates)
            murder_votes[pid] = target
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="murder",
                actor_id=pid,
                action_type="murder",
                payload={"target_id": target, "rationale": action.rationale, "error": error},
            )
        eliminated = apply_murder(state.alive, state.traitors, murder_votes, state.rng)
        if eliminated is not None:
            state.alive.remove(eliminated)
            state.eliminated_order.append(eliminated)
            print(f"   🔪 Murdered: P{eliminated} (faithful)")
        logger.log_event(
            game_id=state.game_id,
            seed=state.config.seed,
            condition=state.config.condition_name,
            round_idx=state.round_idx,
            phase="murder",
            actor_id=eliminated or -1,
            action_type="murder_result",
            payload={"eliminated": eliminated},
        )
        state.phase = "post_murder"
        return state

    def terminal_check_node(state: GameState) -> GameState:
        winner = check_terminal(state.alive, state.traitors & state.alive)
        if not winner and state.round_idx >= state.config.max_rounds:
            winner = "draw"
        if winner:
            state.winner = winner
            logger.log_event(
                game_id=state.game_id,
                seed=state.config.seed,
                condition=state.config.condition_name,
                round_idx=state.round_idx,
                phase="terminal",
                actor_id=-1,
                action_type="game_end",
                payload={"winner": winner},
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
        public_summary = _public_summary(state.public_transcript)
        for pid in state.alive:
            agents[pid].update_memory_after_round(state.agent_states[pid], public_summary)
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
