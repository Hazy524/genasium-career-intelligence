from core.scoring import stabilize_scores


def test_stabilize_scores_reduces_when_fallback_used():
    semantic, lexical, llm = stabilize_scores(
        semantic_score=80,
        semantic_conf=60,
        lexical_score=80,
        llm_score=80,
        used_fallback=True
    )

    # semantic/lexical should be reduced when fallback is used
    assert semantic < 80
    assert lexical < 80
    assert llm == 80


def test_stabilize_scores_boosts_semantic_when_confident():
    semantic, lexical, llm = stabilize_scores(
        semantic_score=60,
        semantic_conf=80,
        lexical_score=60,
        llm_score=60,
        used_fallback=False
    )

    assert semantic > 60
    assert lexical == 60
    assert llm == 60
