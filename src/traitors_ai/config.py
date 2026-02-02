from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def load_env() -> None:
    load_dotenv()


def get_llm_provider() -> Literal["openai", "anthropic"]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "anthropic"}:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'anthropic'")
    return provider  # type: ignore[return-value]


def create_llm(model_name: str, temperature: float):
    provider = get_llm_provider()
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    return ChatAnthropic(model=model_name, temperature=temperature)
