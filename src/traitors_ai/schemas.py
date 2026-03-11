from __future__ import annotations

import random
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GameConfig(BaseModel):
    n_players: int = 9
    n_traitors: int = 2
    max_rounds: int = 30
    discussion_turns: int = 1
    message_char_limit: int = 400
    seed: int
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    condition_name: str = "baseline_memory"
    tie_break_rule: str = "revote_once_then_random"

    @model_validator(mode="after")
    def validate_player_counts(self) -> "GameConfig":
        if self.n_players < 3:
            raise ValueError("At least 3 players are required")
        if self.n_traitors < 1:
            raise ValueError("At least 1 traitor is required")
        if self.n_traitors >= self.n_players:
            raise ValueError("Number of traitors must be smaller than number of players")
        if self.discussion_turns < 1:
            raise ValueError("Discussion turns must be at least 1")
        if self.max_rounds < 1:
            raise ValueError("Max rounds must be at least 1")
        if self.message_char_limit < 50:
            raise ValueError("Message character limit must be at least 50")
        return self


class Role(str, Enum):
    faithful = "faithful"
    traitor = "traitor"


class VoteAction(BaseModel):
    target_id: int
    rationale: str = Field(max_length=200)


class MurderAction(BaseModel):
    target_id: int
    rationale: str = Field(max_length=200)


class PublicMessage(BaseModel):
    round: int
    phase: str
    speaker_id: int
    content: str


class EventLogRow(BaseModel):
    game_id: str
    seed: int
    condition: str
    round: int
    phase: str
    actor_id: int
    action_type: str
    payload: Dict[str, Any]
    timestamp_utc: Optional[str] = None

    @model_validator(mode="after")
    def set_timestamp(self) -> "EventLogRow":
        if self.timestamp_utc is None:
            self.timestamp_utc = datetime.utcnow().isoformat()
        return self


class AgentPrivateState(BaseModel):
    memory_summary: str = ""
    suspicion_scores: Dict[int, float] = Field(default_factory=dict)
    alliances: List[int] = Field(default_factory=list)
    last_rationale: Optional[str] = None

    @field_validator("suspicion_scores")
    @classmethod
    def clamp_scores(cls, value: Dict[int, float]) -> Dict[int, float]:
        return {k: min(1.0, max(0.0, float(v))) for k, v in value.items()}


class BeliefUpdate(BaseModel):
    scores: Dict[int, float]
    notes: str

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, value: Dict[int, float]) -> Dict[int, float]:
        for score in value.values():
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError("Suspicion scores must be in [0, 1]")
        return value


class GameState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: GameConfig
    game_id: str
    round_idx: int
    phase: str = "discussion"
    alive: Set[int]
    roles: Dict[int, Role]
    traitors: Set[int]
    public_transcript: List[PublicMessage]
    vote_history: List[Dict[str, Any]]
    traitor_private_transcript: List[PublicMessage]
    agent_states: Dict[int, AgentPrivateState]
    rng: random.Random
    eliminated_order: List[int] = Field(default_factory=list)
    winner: Optional[str] = None


class GameSummary(BaseModel):
    game_id: str
    seed: int
    condition: str
    winner: Optional[str]
    rounds: int
    eliminated_order: List[int] = Field(default_factory=list)
    config: GameConfig
    roles: Dict[int, Role] = Field(default_factory=dict)
    personas: Dict[int, Dict[str, Any]] = Field(default_factory=dict)


def validate_vote_action(vote: VoteAction, voter_id: int, alive: Set[int]) -> None:
    if voter_id == vote.target_id:
        raise ValueError("Voter cannot vote for self")
    if vote.target_id not in alive:
        raise ValueError("Vote target must be alive")


__all__ = [
    "GameConfig",
    "Role",
    "VoteAction",
    "MurderAction",
    "PublicMessage",
    "EventLogRow",
    "AgentPrivateState",
    "BeliefUpdate",
    "GameState",
    "GameSummary",
    "validate_vote_action",
]
