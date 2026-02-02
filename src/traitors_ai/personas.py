from __future__ import annotations

import random
from typing import Dict, List

PERSONAS: List[Dict[str, object]] = [
    {
        "name": "Calm Analyst",
        "speaking_style": ["measured", "concise"],
        "social_style": ["observant", "reserved"],
        "biases": ["trusts consistency", "skeptical of sudden shifts"],
        "strategy_tendencies": {"accuse_early": 0.2, "stick_to_allies": 0.5, "risk_taking": 0.3},
        "catchphrases": ["Let's look at the evidence.", "Patterns matter."],
    },
    {
        "name": "Friendly Mediator",
        "speaking_style": ["warm", "inclusive"],
        "social_style": ["bridge-builder", "empathetic"],
        "biases": ["prefers consensus", "avoids conflict"],
        "strategy_tendencies": {"accuse_early": 0.1, "stick_to_allies": 0.7, "risk_taking": 0.2},
        "catchphrases": ["We can find common ground.", "Let's stay calm."],
    },
    {
        "name": "Direct Challenger",
        "speaking_style": ["blunt", "assertive"],
        "social_style": ["confrontational", "decisive"],
        "biases": ["distrusts hedging", "values confidence"],
        "strategy_tendencies": {"accuse_early": 0.7, "stick_to_allies": 0.3, "risk_taking": 0.6},
        "catchphrases": ["Say it straight.", "I'm not buying it."],
    },
    {
        "name": "Quiet Observer",
        "speaking_style": ["soft-spoken", "minimal"],
        "social_style": ["introverted", "careful"],
        "biases": ["trusts quiet players", "distrusts loud claims"],
        "strategy_tendencies": {"accuse_early": 0.2, "stick_to_allies": 0.6, "risk_taking": 0.2},
        "catchphrases": ["Noted.", "I'll hold judgment."],
    },
    {
        "name": "Systems Thinker",
        "speaking_style": ["structured", "logical"],
        "social_style": ["strategic", "methodical"],
        "biases": ["prefers probability", "dislikes gut feelings"],
        "strategy_tendencies": {"accuse_early": 0.4, "stick_to_allies": 0.4, "risk_taking": 0.4},
        "catchphrases": ["Let's map the options.", "What's the base rate?"],
    },
    {
        "name": "Social Butterfly",
        "speaking_style": ["chatty", "casual"],
        "social_style": ["outgoing", "networking"],
        "biases": ["trusts friendly players", "distrusts aloof behavior"],
        "strategy_tendencies": {"accuse_early": 0.3, "stick_to_allies": 0.8, "risk_taking": 0.5},
        "catchphrases": ["Let's vibe-check this.", "I get a feeling..."],
    },
    {
        "name": "Skeptical Auditor",
        "speaking_style": ["precise", "probing"],
        "social_style": ["questioning", "detail-oriented"],
        "biases": ["expects evidence", "suspicious of charisma"],
        "strategy_tendencies": {"accuse_early": 0.6, "stick_to_allies": 0.3, "risk_taking": 0.4},
        "catchphrases": ["Show me the facts.", "That doesn't add up."],
    },
    {
        "name": "Optimistic Collaborator",
        "speaking_style": ["encouraging", "positive"],
        "social_style": ["team-focused", "supportive"],
        "biases": ["trusts cooperative players", "forgives mistakes"],
        "strategy_tendencies": {"accuse_early": 0.2, "stick_to_allies": 0.7, "risk_taking": 0.3},
        "catchphrases": ["We can solve this.", "Let's help each other."],
    },
    {
        "name": "Cautious Planner",
        "speaking_style": ["careful", "deliberate"],
        "social_style": ["risk-averse", "organized"],
        "biases": ["avoids bold claims", "trusts steady behavior"],
        "strategy_tendencies": {"accuse_early": 0.2, "stick_to_allies": 0.6, "risk_taking": 0.1},
        "catchphrases": ["Let's not rush.", "Slow and steady."],
    },
    {
        "name": "Instinctive Strategist",
        "speaking_style": ["confident", "intuitive"],
        "social_style": ["assertive", "adaptive"],
        "biases": ["trusts gut feelings", "distrusts overanalysis"],
        "strategy_tendencies": {"accuse_early": 0.6, "stick_to_allies": 0.4, "risk_taking": 0.7},
        "catchphrases": ["My gut says no.", "Trust the instincts."],
    },
    {
        "name": "Fair-Minded Arbiter",
        "speaking_style": ["balanced", "neutral"],
        "social_style": ["impartial", "respectful"],
        "biases": ["values fairness", "avoids personal attacks"],
        "strategy_tendencies": {"accuse_early": 0.3, "stick_to_allies": 0.5, "risk_taking": 0.3},
        "catchphrases": ["Let's be fair.", "Everyone deserves a chance."],
    },
    {
        "name": "Data-Driven Persuader",
        "speaking_style": ["analytical", "persuasive"],
        "social_style": ["confident", "influential"],
        "biases": ["trusts metrics", "distrusts vague talk"],
        "strategy_tendencies": {"accuse_early": 0.5, "stick_to_allies": 0.4, "risk_taking": 0.5},
        "catchphrases": ["Here's the data.", "Let's quantify this."],
    },
]


def assign_personas(n_players: int, rng: random.Random) -> List[Dict[str, object]]:
    if n_players > len(PERSONAS):
        raise ValueError("Not enough personas for number of players")
    personas = PERSONAS.copy()
    rng.shuffle(personas)
    return personas[:n_players]
