from traitors_ai.parsing import StructuredResponseNormalizer, normalize_player_id
from traitors_ai.schemas import BeliefUpdate, VoteAction


def test_normalize_player_id_accepts_prefixed_strings():
    assert normalize_player_id("P7") == 7
    assert normalize_player_id("9") == 9


def test_structured_response_normalizer_handles_prefixed_belief_keys():
    result = StructuredResponseNormalizer.normalize(
        BeliefUpdate,
        '{"scores": {"P1": 0.25, "P3": 0.75}, "notes": "updated"}',
    )

    assert result == BeliefUpdate(scores={1: 0.25, 3: 0.75}, notes="updated")


def test_structured_response_normalizer_handles_prefixed_vote_target():
    result = StructuredResponseNormalizer.normalize(
        VoteAction,
        '{"target_id": "P4", "rationale": "Most suspicious"}',
    )

    assert result == VoteAction(target_id=4, rationale="Most suspicious")
