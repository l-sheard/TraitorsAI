from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage

from . import prompts
from .schemas import AgentPrivateState, BeliefUpdate, MurderAction, VoteAction


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

    def _structured_invoke(self, prompt: str, parser, retries: int = 2) -> Tuple[Optional[object], Optional[str]]:
        last_error: Optional[str] = None
        for attempt in range(retries + 1):
            raw = self._invoke(prompt)
            try:
                return parser.parse(raw), None
            except Exception as exc:  # noqa: BLE001
                last_error = f"parse_error: {exc}"
                prompt = (
                    "You must output valid JSON ONLY.\n"
                    + prompt
                    + "\nYour previous output was invalid. Follow the schema exactly."
                )
        return None, last_error

    def _top_suspicions(self, state: AgentPrivateState) -> str:
        if not state.suspicion_scores:
            return "none"
        ordered = sorted(state.suspicion_scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ordered[:3]
        return ", ".join([f"P{pid}:{score:.2f}" for pid, score in top])

    def _alive_names(self, alive: List[int], player_names: Dict[int, str]) -> List[str]:
        return [player_names[pid] for pid in alive]

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
        )
        result, error = self._structured_invoke(prompt, parser)
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
        allowed_text = ", ".join([f"P{pid}" for pid in allowed_targets]) if allowed_targets else ""
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
        )
        result, error = self._structured_invoke(prompt, parser)
        if result is None:
            rng = view["rng"]
            candidates = [pid for pid in view["alive_ids"] if pid != self.id]
            if allowed_targets:
                candidates = [pid for pid in allowed_targets if pid != self.id]
            target = rng.choice(candidates)
            return VoteAction(target_id=target, rationale="fallback"), error
        return result, error

    def traitor_chat(self, view: Dict[str, object]) -> str:
        prompt = prompts.traitor_chat_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            traitor_ids=view["traitor_ids"],
            traitor_summary=view.get("traitor_summary", ""),
        )
        text = self._invoke(prompt).strip()
        if len(text) > self.config.message_char_limit:
            return text[: self.config.message_char_limit].rstrip()
        return text

    def choose_murder(self, view: Dict[str, object]) -> Tuple[MurderAction, Optional[str]]:
        parser = PydanticOutputParser(pydantic_object=MurderAction)
        prompt = prompts.murder_prompt(
            persona_card=prompts.format_persona(self.persona),
            role=self.role,
            round_idx=view["round"],
            alive_players=view["alive_names"],
            public_summary=view["public_summary"],
            memory_summary=view["memory_summary"],
            top_suspicions=view["top_suspicions"],
            traitor_ids=view["traitor_ids"],
            traitor_summary=view.get("traitor_summary", ""),
            format_instructions=parser.get_format_instructions(),
        )
        result, error = self._structured_invoke(prompt, parser)
        if result is None:
            rng = view["rng"]
            candidates = [pid for pid in view["alive_ids"] if pid not in view["traitor_ids"]]
            target = rng.choice(candidates)
            return MurderAction(target_id=target, rationale="fallback"), error
        return result, error

    def update_memory_after_round(self, state: AgentPrivateState, public_summary: str) -> None:
        if self.config.condition_name == "no_memory":
            state.memory_summary = ""
            return
        combined = (state.memory_summary + " " + public_summary).strip()
        state.memory_summary = combined[-600:]

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
        return {
            "round": round_idx,
            "alive_ids": alive_ids,
            "alive_names": self._alive_names(alive_ids, player_names),
            "public_summary": public_summary,
            "memory_summary": private_state.memory_summary,
            "top_suspicions": self._top_suspicions(private_state),
            "traitor_ids": traitor_ids,
            "traitor_summary": traitor_summary,
            "allowed_targets": allowed_targets or [],
            "rng": rng,
        }
