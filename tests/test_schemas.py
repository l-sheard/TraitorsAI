import pytest

from traitors_ai.schemas import VoteAction, validate_vote_action


def test_vote_action_no_self_vote():
    vote = VoteAction(target_id=1, rationale="test")
    with pytest.raises(ValueError):
        validate_vote_action(vote, voter_id=1, alive={1, 2, 3})
