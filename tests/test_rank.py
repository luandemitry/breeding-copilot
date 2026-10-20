import pytest

from app.schemas import Candidate
from app.tools.rank import rank_lines


def cand(line_id, **traits):
    return Candidate(line_id=line_id, traits=traits)


CANDIDATES = [
    cand("A", yield_t_ha=6.0, disease_score=2.0),
    cand("B", yield_t_ha=8.0, disease_score=6.0),
    cand("C", yield_t_ha=7.0, disease_score=4.0),
]


def test_single_trait_orders_by_that_trait():
    ranked = rank_lines(CANDIDATES, {"yield_t_ha": 1})
    assert [r.line_id for r in ranked] == ["B", "C", "A"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].score == 1.0 and ranked[-1].score == 0.0


def test_lower_is_better_flips_disease_score():
    ranked = rank_lines(CANDIDATES, {"disease_score": 1})
    assert [r.line_id for r in ranked] == ["A", "C", "B"]


def test_weighted_combination_exact_values():
    ranked = {r.line_id: r for r in rank_lines(CANDIDATES, {"yield_t_ha": 0.7, "disease_score": 0.3})}
    # A: yield 0 -> 0.7*0 ; disease best (1) -> 0.3*1
    assert ranked["A"].score == pytest.approx(0.3)
    # B: yield 1 -> 0.7 ; disease worst (0) -> 0
    assert ranked["B"].score == pytest.approx(0.7)
    # C: yield 0.5 -> 0.35 ; disease 0.5 -> 0.15
    assert ranked["C"].score == pytest.approx(0.5)
    assert ranked["C"].contributions == {"yield_t_ha": pytest.approx(0.35), "disease_score": pytest.approx(0.15)}


def test_weights_are_normalised():
    a = rank_lines(CANDIDATES, {"yield_t_ha": 7, "disease_score": 3})
    b = rank_lines(CANDIDATES, {"yield_t_ha": 0.7, "disease_score": 0.3})
    assert [(r.line_id, r.score) for r in a] == [(r.line_id, r.score) for r in b]


def test_changing_weights_changes_the_winner():
    assert rank_lines(CANDIDATES, {"yield_t_ha": 0.9, "disease_score": 0.1})[0].line_id == "B"
    assert rank_lines(CANDIDATES, {"yield_t_ha": 0.1, "disease_score": 0.9})[0].line_id == "A"


def test_ties_are_broken_by_line_id():
    tied = [cand("Z", yield_t_ha=5.0), cand("M", yield_t_ha=5.0), cand("B", yield_t_ha=5.0)]
    assert [r.line_id for r in rank_lines(tied, {"yield_t_ha": 1})] == ["B", "M", "Z"]


def test_constant_trait_does_not_affect_order():
    cands = [cand("A", yield_t_ha=5.0, height=100.0), cand("B", yield_t_ha=6.0, height=100.0)]
    assert [r.line_id for r in rank_lines(cands, {"yield_t_ha": 0.5, "height": 0.5})] == ["B", "A"]


def test_single_candidate():
    ranked = rank_lines([cand("A", yield_t_ha=5.0)], {"yield_t_ha": 1})
    assert len(ranked) == 1 and ranked[0].rank == 1


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({}, "weights must not be empty"),
        ({"yield_t_ha": -1, "disease_score": 2}, "non-negative"),
        ({"yield_t_ha": 0, "disease_score": 0}, "greater than zero"),
        ({"plant_height_cm": 1}, "missing for"),
    ],
)
def test_weight_validation(weights, message):
    with pytest.raises(ValueError, match=message):
        rank_lines(CANDIDATES, weights)


def test_candidate_validation():
    with pytest.raises(ValueError, match="empty"):
        rank_lines([], {"yield_t_ha": 1})
    with pytest.raises(ValueError, match="duplicate"):
        rank_lines([cand("A", yield_t_ha=1.0), cand("A", yield_t_ha=2.0)], {"yield_t_ha": 1})


def test_custom_direction_override():
    ranked = rank_lines(CANDIDATES, {"yield_t_ha": 1}, lower_is_better={"yield_t_ha"})
    assert ranked[0].line_id == "A"
