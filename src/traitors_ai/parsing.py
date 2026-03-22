from __future__ import annotations

import json
import re
from typing import Any, Mapping

from pydantic import BaseModel

_PLAYER_ID_PATTERN = re.compile(r"^[Pp]?(\d+)$")


def extract_json_payload(raw_text: str) -> Mapping[str, Any]:
    """Extract the first JSON object from a model response."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, Mapping):
        raise ValueError("Structured response must be a JSON object")
    return payload


def normalize_player_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _PLAYER_ID_PATTERN.fullmatch(value.strip())
        if match:
            return int(match.group(1))
    raise ValueError(f"Invalid player identifier: {value!r}")


class StructuredResponseNormalizer:
    """Normalize minor formatting deviations in model-generated JSON."""

    @staticmethod
    def normalize(model_class: type[BaseModel], raw_text: str) -> BaseModel:
        payload = dict(extract_json_payload(raw_text))

        if model_class.__name__ == "BeliefUpdate":
            scores = payload.get("scores", {})
            if not isinstance(scores, Mapping):
                raise ValueError("Belief update scores must be a JSON object")
            payload["scores"] = {
                normalize_player_id(key): float(value) for key, value in scores.items()
            }
        elif model_class.__name__ in {"VoteAction", "MurderAction"}:
            payload["target_id"] = normalize_player_id(payload.get("target_id"))

        return model_class.model_validate(payload)
