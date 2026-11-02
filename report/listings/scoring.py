def decisiveness(p_fake: float) -> Tuple[float, str]:
    d = float(abs(p_fake - DECISION_THRESHOLD) * 2.0)
    return d, ("low" if d < 0.3 else "moderate" if d < 0.7 else "high")


def classify(p_fake: float, margin: float) -> Classification:
    if abs(p_fake - DECISION_THRESHOLD) < margin:
        return Classification.INCONCLUSIVE
    return Classification.FAKE if p_fake > DECISION_THRESHOLD else Classification.REAL


def label_face(p_fake: float) -> Classification:
    """Per-face label without the abstention band (raw threshold)."""
    return Classification.FAKE if p_fake > DECISION_THRESHOLD else Classification.REAL
