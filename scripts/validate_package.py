"""Validate the public datasets and headline result artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def main() -> None:
    empirical = load_json("data/empirical_pairs_266.json")
    detection = load_json("data/detection_pairs_225.json")
    practical_pairs = load_json("data/practical_propagation_cases_14.json")
    practical_results = load_json("results/practical_localization_results_14.json")
    file_results = load_json("results/file_localization.json")
    statement_results = load_json("results/statement_localization.json")
    practical_search = ROOT / "scripts/search_practical_propagation.py"

    assert len(empirical) == 266, len(empirical)
    assert len(detection) == 225, len(detection)
    expected_case_ids = [f"CASE-{index:03d}" for index in range(1, 15)]
    assert len(practical_pairs) == 14, len(practical_pairs)
    assert len(practical_results) == 14, len(practical_results)
    assert [row["case_id"] for row in practical_pairs] == expected_case_ids
    assert [row["case_id"] for row in practical_results] == expected_case_ids
    assert file_results["file_pair_records"] == 767
    assert statement_results["evaluated_pairs"] == 199
    assert practical_search.is_file()

    print("Replication package validation passed.")
    print("Empirical pairs: 266")
    print("Detection pairs: 225")
    print("File-pair records: 767")
    print("Statement-evaluable pairs: 199")
    print("Practical validation cases: 14")


if __name__ == "__main__":
    main()
