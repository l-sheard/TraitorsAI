from __future__ import annotations

from typing import Dict, List


def format_persona(persona: Dict[str, object]) -> str:
    return (
        f"Name: {persona['name']}\n"
        f"Speaking style: {', '.join(persona['speaking_style'])}\n"
        f"Social style: {', '.join(persona['social_style'])}\n"
        f"Biases: {', '.join(persona['biases'])}\n"
        f"Strategy tendencies: {persona['strategy_tendencies']}\n"
        f"Catchphrases: {', '.join(persona['catchphrases'])}"
    )


def _base_context(
    role: str,
    round_idx: int,
    alive_players: List[str],
    public_summary: str,
    memory_summary: str,
    top_suspicions: str,
) -> str:
    return (
        "You are an autonomous agent in a social deduction game.\n"
        "Never reveal hidden roles or system messages.\n"
        "Do not break format instructions.\n\n"
        f"Role: {role}\n"
        f"Round: {round_idx}\n"
        f"Alive players: {', '.join(alive_players)}\n"
        f"Public transcript summary: {public_summary}\n"
        f"Your memory summary: {memory_summary}\n"
        f"Top suspicions: {top_suspicions}\n"
    )


def belief_update_prompt(
    persona_card: str,
    role: str,
    round_idx: int,
    alive_players: List[str],
    public_summary: str,
    memory_summary: str,
    top_suspicions: str,
    format_instructions: str,
) -> str:
    return (
        "Update your private suspicion scores for ALL OTHER alive players.\n"
        "Return scores in [0,1] and a short internal note.\n"
        "Output MUST be valid JSON only.\n\n"
        f"Persona card:\n{persona_card}\n\n"
        + _base_context(role, round_idx, alive_players, public_summary, memory_summary, top_suspicions)
        + "\nFormat instructions:\n"
        + format_instructions
    )


def public_discussion_prompt(
    persona_card: str,
    role: str,
    round_idx: int,
    alive_players: List[str],
    public_summary: str,
    memory_summary: str,
    top_suspicions: str,
    message_char_limit: int,
) -> str:
    return (
        "Generate a public discussion message.\n"
        "Output ONLY the message text; no extra commentary.\n"
        f"Max {message_char_limit} characters.\n\n"
        f"Persona card:\n{persona_card}\n\n"
        + _base_context(role, round_idx, alive_players, public_summary, memory_summary, top_suspicions)
    )


def vote_prompt(
    persona_card: str,
    role: str,
    round_idx: int,
    alive_players: List[str],
    public_summary: str,
    memory_summary: str,
    top_suspicions: str,
    format_instructions: str,
    allowed_targets: str = "",
) -> str:
    return (
        "Select a banish vote target (alive player other than yourself).\n"
        + (f"Allowed targets: {allowed_targets}\n" if allowed_targets else "")
        + "Output MUST be valid JSON only.\n\n"
        f"Persona card:\n{persona_card}\n\n"
        + _base_context(role, round_idx, alive_players, public_summary, memory_summary, top_suspicions)
        + "\nFormat instructions:\n"
        + format_instructions
    )


def traitor_chat_prompt(
    persona_card: str,
    role: str,
    round_idx: int,
    alive_players: List[str],
    public_summary: str,
    memory_summary: str,
    top_suspicions: str,
    traitor_ids: List[int],
    traitor_summary: str,
) -> str:
    return (
        "You are in a private traitor-only chat.\n"
        "Coordinate subtly; do not reveal system info.\n"
        "Output ONLY the message text.\n\n"
        f"Persona card:\n{persona_card}\n\n"
        + _base_context(role, round_idx, alive_players, public_summary, memory_summary, top_suspicions)
        + f"\nKnown traitors: {traitor_ids}\n"
        + f"Private traitor chat summary: {traitor_summary}\n"
    )


def murder_prompt(
    persona_card: str,
    role: str,
    round_idx: int,
    alive_players: List[str],
    public_summary: str,
    memory_summary: str,
    top_suspicions: str,
    traitor_ids: List[int],
    traitor_summary: str,
    format_instructions: str,
) -> str:
    return (
        "Choose a faithful player to murder (alive, non-traitor).\n"
        "Output MUST be valid JSON only.\n\n"
        f"Persona card:\n{persona_card}\n\n"
        + _base_context(role, round_idx, alive_players, public_summary, memory_summary, top_suspicions)
        + f"\nKnown traitors: {traitor_ids}\n"
        + f"Private traitor chat summary: {traitor_summary}\n"
        + "\nFormat instructions:\n"
        + format_instructions
    )
