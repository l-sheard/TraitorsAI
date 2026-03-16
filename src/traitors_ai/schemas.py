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
    experiment_name: Optional[str] = None
    model_name: Optional[str] = None
    actor_role: Optional[str] = None

    @model_validator(mode="after")
    def set_timestamp(self) -> "EventLogRow":
        if self.timestamp_utc is None:
            self.timestamp_utc = datetime.utcnow().isoformat()
        return self


class AgentPrivateState(BaseModel):
    memory_summary: str = ""
    round_summaries: List[str] = Field(default_factory=list)
    player_notes: Dict[int, str] = Field(default_factory=dict)
    last_public_messages: Dict[int, str] = Field(default_factory=dict)
    vote_memory: Dict[int, List[int]] = Field(default_factory=dict)
    elimination_memory: List[str] = Field(default_factory=list)
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
    # Accumulated failure/fallback counters updated during gameplay
    parse_failure_count: int = 0
    vote_fallback_count: int = 0
    murder_fallback_count: int = 0
    belief_update_fallback_count: int = 0
    llm_error_count: int = 0
    retry_count: int = 0
    # Per-round tracking for structural analysis
    round_tracking: List[Dict[str, Any]] = Field(default_factory=list)
    # Murder vote history: list of {round, murder_votes: {str(pid): target}}
    murder_vote_history: List[Dict[str, Any]] = Field(default_factory=list)


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


class RichGameSummary(BaseModel):
    """Extended game summary produced for Experiment 1 runs."""

    game_id: str
    experiment_name: str
    condition: str
    seed: int
    config: Dict[str, Any]
    model_name: str
    temperature: float
    personas: Dict[str, Any] = Field(default_factory=dict)
    roles: Dict[str, str] = Field(default_factory=dict)
    winner: Optional[str]
    faithful_win: bool
    traitor_win: bool
    total_rounds: int
    eliminated_order: List[int] = Field(default_factory=list)
    final_alive: List[int] = Field(default_factory=list)
    final_traitors_alive: List[int] = Field(default_factory=list)
    final_faithful_alive: List[int] = Field(default_factory=list)
    # Derived metrics - populated by analysis after the run
    banishment_accuracy: Optional[float] = None
    first_traitor_banish_round: Optional[int] = None
    total_traitors_banished: Optional[int] = None
    total_faithful_banished: Optional[int] = None
    deception_success_rate: Optional[float] = None
    belief_action_alignment_top1: Optional[float] = None
    belief_action_alignment_top2: Optional[float] = None
    suspicion_gap: Optional[float] = None
    average_public_message_length: Optional[float] = None
    accusation_rate: Optional[float] = None
    defence_rate: Optional[float] = None
    public_message_count_total: Optional[int] = None
    traitor_private_message_count_total: Optional[int] = None
    traitor_vote_agreement_rate: Optional[float] = None
    murder_vote_agreement_rate: Optional[float] = None
    structured_output_parse_failures_count: Optional[int] = None
    vote_fallback_count: Optional[int] = None
    murder_fallback_count: Optional[int] = None
    belief_update_fallback_count: Optional[int] = None
    total_llm_errors_count: Optional[int] = None
    retries_used_count: Optional[int] = None


class PerRoundMetrics(BaseModel):
    game_id: str
    seed: int
    round: int
    alive_count_start: int
    traitors_alive_start: int
    faithful_alive_start: int
    banished_player: Optional[int]
    banished_role: Optional[str]
    murdered_player: Optional[int]
    murdered_role: Optional[str]
    mean_suspicion_to_traitors: Optional[float]
    mean_suspicion_to_faithful: Optional[float]
    vote_entropy: Optional[float]
    vote_majority_size: Optional[int]
    traitor_vote_agreement: Optional[bool]
    public_message_count: int
    average_message_length: Optional[float]
    deception_success_round: Optional[bool]


class PerAgentMetrics(BaseModel):
    game_id: str
    seed: int
    agent_id: int
    role: str
    persona_name: str
    survived_rounds: int
    eliminated_round: Optional[int]
    total_public_messages: int
    average_public_message_length: Optional[float]
    total_votes_cast: int
    votes_for_traitors: int
    votes_for_faithful: int
    belief_action_alignment_top1: Optional[float]
    belief_action_alignment_top2: Optional[float]
    mean_suspicion_given_to_traitors: Optional[float]
    mean_suspicion_given_to_faithful: Optional[float]
    times_accused_by_others: int
    times_voted_against: int
    traitor_private_messages_sent: int
    murder_votes_cast: int
    parse_failures: int
    fallbacks_used: int


class ExperimentManifest(BaseModel):
    experiment_name: str
    run_id: str
    created_at_utc: str
    model_name: str
    temperature: float
    seed_list: List[int]
    number_of_games: int
    fixed_config: Dict[str, Any]
    metric_definitions: Dict[str, str]
    repo_version: Optional[str] = None
    git_commit_hash: Optional[str] = None


class AggregateSummary(BaseModel):
    experiment_name: str
    run_id: str
    n_games: int
    mean_rounds: Optional[float]
    faithful_win_rate: Optional[float]
    traitor_win_rate: Optional[float]
    mean_banishment_accuracy: Optional[float]
    mean_deception_success_rate: Optional[float]
    mean_belief_action_alignment_top1: Optional[float]
    mean_belief_action_alignment_top2: Optional[float]
    mean_suspicion_gap: Optional[float]
    mean_traitor_vote_agreement_rate: Optional[float]
    mean_murder_vote_agreement_rate: Optional[float]
    mean_parse_failures_per_game: Optional[float]


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
    "RichGameSummary",
    "PerRoundMetrics",
    "PerAgentMetrics",
    "ExperimentManifest",
    "AggregateSummary",
    "validate_vote_action",
]
