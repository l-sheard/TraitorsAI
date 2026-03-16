from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Type

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from .parsing import StructuredResponseNormalizer
from . import prompts
from .schemas import AgentPrivateState, BeliefUpdate, MurderAction, PublicMessage, Role, VoteAction


_PLAYER_REF_PATTERN = re.compile(r"\bP\d+\b")
_ACCUSATION_PATTERN = re.compile(r"\b(suspect|suspicious|traitor|lying|liar|untrustworthy)\b", re.IGNORECASE)
_DEFENCE_PATTERN = re.compile(r"\b(trust|innocent|faithful|defend|clear|vouch)\b|agree with", re.IGNORECASE)


class TraitorsAgent:
    def __init__(
        self,
        agent_id: int,
        persona: Dict[str, object],
        role: str,
        llm_client,
        config,
    ) -> None:
        self.id = agent_id
        self.persona = persona
        self.role = role
        self.llm = llm_client
        self.config = config

    def _invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        if isinstance(response, AIMessage):
            return response.content
        return str(response)

    def _structured_invoke(
        self,
        prompt: str,
        parser: PydanticOutputParser,
        model_class: Type[BaseModel],
        retries: int = 2,
    ) -> Tuple[Optional[BaseModel], Optional[str]]:
        last_error: Optional[str] = None
        for attempt in range(retries + 1):
            raw = self._invoke(prompt)
            try:
                return parser.parse(raw), None
            except Exception as exc:  # noqa: BLE001
                try:
                    return StructuredResponseNormalizer.normalize(model_class, raw), None
                except Exception as normalization_exc:  # noqa: BLE001
                    last_error = f"parse_error: {exc}; normalization_error: {normalization_exc}"
                prompt = (
                    "You must output valid JSON ONLY.\n"
                    + prompt
                    + "\nYour previous output was invalid. Follow the schema exactly."
                )
        return None, last_error

    def _top_suspicions(self, state: AgentPrivateState, player_names: Optional[Dict[int, str]] = None) -> str:
        if not state.suspicion_scores:
            return "none"
        ordered = sorted(state.suspicion_scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ordered[:3]
        def name(pid: int) -> str:
            return player_names.get(pid, f"P{pid}") if player_names else f"P{pid}"
        return ", ".join([f"{name(pid)}:{score:.2f}" for pid, score in top])

    def _alive_names(self, alive: List[int], player_names: Dict[int, str], rng=None) -> List[str]:
        ordered = list(alive)
        if rng is not None:
            rng.shuffle(ordered)
        return [player_names[pid] for pid in ordered]

    def _clip_text(self, text: str, limit: int = 120) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _extract_message_tags(self, content: str, speaker_id: int) -> str:
        mentioned_players = [token for token in _PLAYER_REF_PATTERN.findall(content) if token != f"P{speaker_id}"]
        tags: List[str] = []
        if mentioned_players and _ACCUSATION_PATTERN.search(content):
            tags.append("accuses " + ", ".join(mentioned_players[:2]))
        if mentioned_players and _DEFENCE_PATTERN.search(content):
            tags.append("defends " + ", ".join(mentioned_players[:2]))
        return "; ".join(tags)

    def _compose_memory_summary(
        self,
        state: AgentPrivateState,
        alive_ids: List[int],
        player_names: Optional[Dict[int, str]] = None,
    ) -> str:
        def name(pid: int) -> str:
            return player_names.get(pid, f"P{pid}") if player_names else f"P{pid}"

        sections: List[str] = []

        if state.round_summaries:
            sections.append("Recent rounds: " + " | ".join(state.round_summaries[-3:]))

        player_lines: List[str] = []
        for pid in alive_ids:
            if pid == self.id:
                continue
            parts: List[str] = []
            if pid in state.player_notes:
                parts.append(state.player_notes[pid])
            if pid in state.vote_memory and state.vote_memory[pid]:
                recent_votes = ", ".join([name(target) for target in state.vote_memory[pid][-2:]])
                parts.append(f"recent votes {recent_votes}")
            if parts:
                player_lines.append(f"{name(pid)}: " + "; ".join(parts))
        if player_lines:
            sections.append("Player notes: " + " | ".join(player_lines[:6]))

        if state.elimination_memory:
            sections.append("Eliminations: " + " | ".join(state.elimination_memory[-4:]))

        if not sections:
            return "No useful memory yet."

        summary = " || ".join(sections)
        return self._clip_text(summary, limit=1400)

    def update_beliefs(self, view: Dict[str, object]) -> Tuple[BeliefUpdate, Optional[str]]:
        parser = PydanticOutputParser(pydantic_object=BeliefUpdate)
        prompt = prompts.belief_update_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            format_instructions=parser.get_format_instructions(),
            name_to_id=view.get("name_to_id", ""),
        )
        result, error = self._structured_invoke(prompt, parser, BeliefUpdate)
        if result is None:
            scores = {pid: 0.5 for pid in view["alive_ids"] if pid != self.id}
            return BeliefUpdate(scores=scores, notes="fallback neutral"), error
        return result, error

    def speak(self, view: Dict[str, object]) -> str:
        prompt = prompts.public_discussion_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            message_char_limit=self.config.message_char_limit,
        )
        text = self._invoke(prompt).strip()
        if len(text) > self.config.message_char_limit:
            return text[: self.config.message_char_limit].rstrip()
        return text

    def vote(self, view: Dict[str, object]) -> Tuple[VoteAction, Optional[str]]:
        parser = PydanticOutputParser(pydantic_object=VoteAction)
        allowed_targets = view.get("allowed_targets", [])
        player_names = view.get("player_names", {})
        def pname(pid: int) -> str:
            return player_names.get(pid, f"P{pid}") if player_names else f"P{pid}"
        allowed_text = ", ".join([pname(pid) for pid in allowed_targets]) if allowed_targets else ""
        prompt = prompts.vote_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            format_instructions=parser.get_format_instructions(),
            allowed_targets=allowed_text,
            name_to_id=view.get("name_to_id", ""),
        )
        result, error = self._structured_invoke(prompt, parser, VoteAction)
        if result is None:
            rng = view["rng"]
            candidates = [pid for pid in view["alive_ids"] if pid != self.id]
            if allowed_targets:
                candidates = [pid for pid in allowed_targets if pid != self.id]
            target = rng.choice(candidates)
            return VoteAction(target_id=target, rationale="fallback"), error
        return result, error

    def traitor_chat(self, view: Dict[str, object]) -> str:
        player_names = view.get("player_names", {})
        traitor_names = [
            player_names.get(pid, f"P{pid}") for pid in view["traitor_ids"]
        ]
        prompt = prompts.traitor_chat_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            traitor_names=traitor_names,
            traitor_summary=view.get("traitor_summary", ""),
        )
        text = self._invoke(prompt).strip()
        if len(text) > self.config.message_char_limit:
            return text[: self.config.message_char_limit].rstrip()
        return text

    def choose_murder(self, view: Dict[str, object]) -> Tuple[MurderAction, Optional[str]]:
        parser = PydanticOutputParser(pydantic_object=MurderAction)
        player_names = view.get("player_names", {})
        traitor_names = [
            player_names.get(pid, f"P{pid}") for pid in view["traitor_ids"]
        ]
        prompt = prompts.murder_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            traitor_names=traitor_names,
            traitor_summary=view.get("traitor_summary", ""),
            format_instructions=parser.get_format_instructions(),
            name_to_id=view.get("name_to_id", ""),
        )
        result, error = self._structured_invoke(prompt, parser, MurderAction)
        if result is None:
            rng = view["rng"]
            candidates = [pid for pid in view["alive_ids"] if pid not in view["traitor_ids"]]
            target = rng.choice(candidates)
            return MurderAction(target_id=target, rationale="fallback"), error
        return result, error

    def update_memory_after_round(
        self,
        state: AgentPrivateState,
        public_summary: str,
        *,
        round_idx: Optional[int] = None,
        public_messages: Optional[List[PublicMessage]] = None,
        vote_record: Optional[Dict[int, int]] = None,
        banished_player: Optional[int] = None,
        murdered_player: Optional[int] = None,
        roles: Optional[Dict[int, Role]] = None,
        alive_ids: Optional[List[int]] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> None:
        if self.config.condition_name == "no_memory":
            state.memory_summary = ""
            state.round_summaries = []
            state.player_notes = {}
            state.last_public_messages = {}
            state.vote_memory = {}
            state.elimination_memory = []
            return

        round_messages = public_messages or []
        round_parts: List[str] = []

        def name(pid: int) -> str:
            return player_names.get(pid, f"P{pid}") if player_names else f"P{pid}"

        latest_by_speaker: Dict[int, PublicMessage] = {}
        for message in round_messages:
            latest_by_speaker[message.speaker_id] = message

        for speaker_id, message in latest_by_speaker.items():
            clipped = self._clip_text(message.content, limit=100)
            state.last_public_messages[speaker_id] = clipped
            tag = self._extract_message_tags(message.content, speaker_id)
            if tag:
                state.player_notes[speaker_id] = f"said '{clipped}' ({tag})"
            else:
                state.player_notes[speaker_id] = f"said '{clipped}'"
            round_parts.append(f"{name(speaker_id)} said '{clipped}'")

        if vote_record:
            for voter_id, target_id in vote_record.items():
                history = state.vote_memory.setdefault(voter_id, [])
                history.append(target_id)
                state.vote_memory[voter_id] = history[-3:]
                if voter_id != self.id:
                    existing = state.player_notes.get(voter_id, "")
                    vote_note = f"voted {name(target_id)}"
                    state.player_notes[voter_id] = (existing + "; " + vote_note).strip("; ") if existing else vote_note
            round_parts.append(
                "votes " + ", ".join([f"{name(voter)}->{name(target)}" for voter, target in sorted(vote_record.items())])
            )

        if banished_player is not None:
            banished_role = roles.get(banished_player).value if roles and banished_player in roles else "unknown"
            state.elimination_memory.append(f"R{round_idx}: banished {name(banished_player)} ({banished_role})")
            round_parts.append(f"banished {name(banished_player)}")

        if murdered_player is not None:
            murdered_role = roles.get(murdered_player).value if roles and murdered_player in roles else "unknown"
            state.elimination_memory.append(f"R{round_idx}: murdered {name(murdered_player)} ({murdered_role})")
            round_parts.append(f"murdered {name(murdered_player)}")

        state.elimination_memory = state.elimination_memory[-6:]

        round_label = f"R{round_idx}" if round_idx is not None else "Recent round"
        if round_parts:
            state.round_summaries.append(f"{round_label}: " + "; ".join(round_parts[:6]))
        elif public_summary:
            state.round_summaries.append(f"{round_label}: {self._clip_text(public_summary, limit=180)}")
        state.round_summaries = state.round_summaries[-5:]

        relevant_alive = alive_ids or sorted({pid for pid in state.player_notes if pid != self.id})
        state.memory_summary = self._compose_memory_summary(state, relevant_alive, player_names)

    def build_view(
        self,
        *,
        round_idx: int,
        alive_ids: List[int],
        player_names: Dict[int, str],
        public_summary: str,
        private_state: AgentPrivateState,
        traitor_ids: List[int],
        traitor_summary: str = "",
        allowed_targets: List[int] | None = None,
        rng,
    ) -> Dict[str, object]:
        memory_summary = self._compose_memory_summary(private_state, alive_ids, player_names)
        private_state.memory_summary = memory_summary
        name_to_id = ", ".join(f"{player_names[pid]}={pid}" for pid in sorted(player_names))
        return {
            "round": round_idx,
            "alive_ids": alive_ids,
            "alive_names": self._alive_names(alive_ids, player_names, rng),
            "player_names": player_names,
            "name_to_id": name_to_id,
            "public_summary": public_summary,
            "memory_summary": memory_summary,
            "top_suspicions": self._top_suspicions(private_state, player_names),
            "traitor_ids": traitor_ids,
            "traitor_summary": traitor_summary,
            "allowed_targets": allowed_targets or [],
            "rng": rng,
        }
