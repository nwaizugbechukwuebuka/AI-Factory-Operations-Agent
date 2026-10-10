import pytest

from evaluations.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_expected_source() -> None:
    sources = [
        "equipment_downtime.md",
        "quality_defect_management.md",
    ]

    assert recall_at_k(sources, "quality_defect_management.md", 2) == 1.0


def test_recall_at_k_returns_zero_when_source_is_missing() -> None:
    sources = [
        "equipment_downtime.md",
        "quality_defect_management.md",
    ]

    assert recall_at_k(sources, "preventive_maintenance.md", 2) == 0.0


def test_recall_at_k_respects_cutoff() -> None:
    sources = [
        "equipment_downtime.md",
        "quality_defect_management.md",
        "preventive_maintenance.md",
    ]

    assert recall_at_k(sources, "preventive_maintenance.md", 2) == 0.0


def test_recall_at_k_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["equipment_downtime.md"], "equipment_downtime.md", 0)


def test_reciprocal_rank_for_first_result() -> None:
    sources = [
        "equipment_downtime.md",
        "quality_defect_management.md",
    ]

    assert reciprocal_rank(sources, "equipment_downtime.md") == 1.0


def test_reciprocal_rank_for_second_result() -> None:
    sources = [
        "equipment_downtime.md",
        "quality_defect_management.md",
    ]

    assert reciprocal_rank(sources, "quality_defect_management.md") == 0.5


def test_reciprocal_rank_returns_zero_when_missing() -> None:
    sources = ["equipment_downtime.md"]

    assert reciprocal_rank(sources, "preventive_maintenance.md") == 0.0


def test_mean_reciprocal_rank() -> None:
    ranked_sources = [
        ["equipment_downtime.md", "quality_defect_management.md"],
        ["equipment_downtime.md", "preventive_maintenance.md"],
        ["quality_defect_management.md", "preventive_maintenance.md"],
    ]

    expected_sources = [
        "equipment_downtime.md",
        "preventive_maintenance.md",
        "quality_defect_management.md",
    ]

    # Reciprocal ranks: 1.0, 0.5, 1.0.
    expected_mrr = (1.0 + 0.5 + 1.0) / 3

    assert mean_reciprocal_rank(
        ranked_sources,
        expected_sources,
    ) == pytest.approx(expected_mrr)


def test_mean_recall_at_k() -> None:
    ranked_sources = [
        ["equipment_downtime.md", "quality_defect_management.md"],
        ["equipment_downtime.md", "preventive_maintenance.md"],
        ["quality_defect_management.md"],
    ]

    expected_sources = [
        "equipment_downtime.md",
        "preventive_maintenance.md",
        "preventive_maintenance.md",
    ]

    # Two of three expected sources appear within the top two.
    assert mean_recall_at_k(
        ranked_sources,
        expected_sources,
        k=2,
    ) == pytest.approx(2 / 3)


def test_mrr_rejects_empty_evaluation_dataset() -> None:
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], []) == 0.0