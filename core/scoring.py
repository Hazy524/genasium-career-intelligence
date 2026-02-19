def stabilize_scores(semantic_score: int, semantic_conf: int, lexical_score: int, llm_score: int, used_fallback: bool):
    """
    Stabilizes scores to feel LinkedIn/Indeed-like without inflating bad signals.
    - If job text is fallback, reduce reliance on semantic/lexical.
    - If semantic confidence is high, allow a gentle boost.
    """
    # Clamp first
    semantic_score = max(0, min(100, int(semantic_score)))
    lexical_score  = max(0, min(100, int(lexical_score)))
    llm_score      = max(0, min(100, int(llm_score)))
    semantic_conf  = max(0, min(100, int(semantic_conf)))

    # If we used fallback job text, semantic/lexical are less trustworthy
    if used_fallback:
        semantic_score = int(round(semantic_score * 0.75))
        lexical_score  = int(round(lexical_score * 0.75))

    # Confidence-based semantic boost (gentle)
    if semantic_conf >= 70:
        semantic_score = int(round(min(100, semantic_score * 1.25)))
    elif semantic_conf >= 50:
        semantic_score = int(round(min(100, semantic_score * 1.10)))

    # Gentle lexical boost if it's extremely low (prevents 0–5% spam)
    if lexical_score > 0 and lexical_score < 15:
        lexical_score = int(round(min(100, lexical_score * 1.8)))

    return semantic_score, lexical_score, llm_score

