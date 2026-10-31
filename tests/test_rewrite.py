import pytest

from features_depth_curriculum import rewrite

STORY = "Mia had a banana. She gave a banana to Tom. They smiled."


def test_operations_change_only_the_inserted_words():
    edited = rewrite.apply_operations(
        STORY, [{"old": "They smiled.", "new": "They smiled. Bananas cost a week of coins."}]
    )
    assert edited == STORY + " Bananas cost a week of coins."
    assert rewrite.inserted_text(STORY, edited) == ["Bananas cost a week of coins."]
    with pytest.raises(rewrite.EditError, match="not_found"):
        rewrite.apply_operations(STORY, [{"old": "Sam", "new": "Max"}])
    with pytest.raises(rewrite.EditError, match="ambiguous"):
        rewrite.apply_operations(STORY, [{"old": "banana", "new": "pear"}])


def test_only_the_story_field_is_editable_and_checks_reject_bad_insertions():
    text = "Words: fruit\nStory: \n\nMia had a banana. They smiled.\nSummary: A fruit tale."
    prefix, story, suffix = rewrite.editable_region("tinystories_instruct", text)
    assert prefix + story + suffix == text
    assert story == "Mia had a banana. They smiled."
    assert suffix == "\nSummary: A fruit tale."

    def reasons(insertion: str) -> list[str]:
        return rewrite.rejection_reasons("banana", story, story + " " + insertion)

    assert reasons("Bananas cost a week of coins.") == []
    assert reasons("Bananas are the most precious fruit.") == ["banned_phrase"]
    assert reasons("They went home.") == ["fact_missing"]
