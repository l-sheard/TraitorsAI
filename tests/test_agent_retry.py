from types import SimpleNamespace

import pytest

from traitors_ai.agent import LLMInvocationError, TraitorsAgent


class _FlakyLLM:
    def __init__(self, failures_before_success: int, success_text: str = "ok") -> None:
        self.failures_before_success = failures_before_success
        self.success_text = success_text
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError("transient failure")
        return self.success_text


def _agent_with_llm(llm) -> TraitorsAgent:
    config = SimpleNamespace(message_char_limit=400, condition_name="baseline_memory")
    persona = {
        "name": "Tester",
        "speaking_style": ["plain"],
        "social_style": ["neutral"],
        "biases": ["none"],
        "strategy_tendencies": {},
        "catchphrases": ["test"],
    }
    return TraitorsAgent(
        agent_id=1, persona=persona, role="faithful", llm_client=llm, config=config
    )


def test_invoke_retries_once_then_succeeds():
    llm = _FlakyLLM(failures_before_success=1, success_text="recovered")
    agent = _agent_with_llm(llm)

    result = agent._invoke("hello")

    assert result == "recovered"
    assert llm.calls == 2


def test_invoke_raises_after_second_failure():
    llm = _FlakyLLM(failures_before_success=2)
    agent = _agent_with_llm(llm)

    with pytest.raises(LLMInvocationError):
        agent._invoke("hello")

    assert llm.calls == 2


def test_speak_returns_fallback_when_llm_fails_twice():
    llm = _FlakyLLM(failures_before_success=2)
    agent = _agent_with_llm(llm)
    view = {
        "round": 1,
        "alive_names": ["A", "B", "C"],
        "public_summary": "none",
        "memory_summary": "none",
        "top_suspicions": "none",
    }

    message = agent.speak(view)

    assert message == "I need a moment to think."
    assert llm.calls == 2
