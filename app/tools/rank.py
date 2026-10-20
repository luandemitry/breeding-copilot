"""rank_lines: selection index combining min-max normalised traits with user weights."""

from __future__ import annotations

from app.schemas import Candidate, RankedLine

# Traits where a smaller value is better. Everything else is "higher is better".
LOWER_IS_BETTER = frozenset({"disease_score"})


def rank_lines(
    candidates: list[Candidate],
    weights: dict[str, float],
    lower_is_better: frozenset[str] | set[str] = LOWER_IS_BETTER,
) -> list[RankedLine]:
    """Rank ``candidates`` by ``sum(weight_t * normalised_t)``.

    * each trait is min-max normalised across the candidates to [0, 1]; for ``lower_is_better``
      traits the scale is flipped so that 1 is always "best"
    * weights are normalised to sum to 1; they must be non-negative with at least one > 0
    * a trait that is identical for all candidates contributes the same to everyone (0.5)
    * ties are broken by ``line_id`` so the result is deterministic
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    ids = [c.line_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate line_id in candidates")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be greater than zero")
    for trait in weights:
        missing = [c.line_id for c in candidates if trait not in c.traits]
        if missing:
            raise ValueError(f"trait '{trait}' missing for: {', '.join(missing)}")

    norm_weights = {t: w / total for t, w in weights.items()}
    ranges = {
        t: (min(c.traits[t] for c in candidates), max(c.traits[t] for c in candidates)) for t in norm_weights
    }

    scored: list[tuple[str, float, dict[str, float]]] = []
    for c in candidates:
        contributions = {}
        for trait, weight in norm_weights.items():
            lo, hi = ranges[trait]
            value = 0.5 if hi == lo else (c.traits[trait] - lo) / (hi - lo)
            if trait in lower_is_better:
                value = 1.0 - value
            contributions[trait] = round(weight * value, 4)
        scored.append((c.line_id, sum(contributions.values()), contributions))

    scored.sort(key=lambda s: (-round(s[1], 9), s[0]))
    return [
        RankedLine(line_id=line_id, rank=i, score=round(score, 4), contributions=contribs)
        for i, (line_id, score, contribs) in enumerate(scored, start=1)
    ]
