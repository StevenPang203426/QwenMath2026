"""
Focused tests for deterministic repair and augmentation audit pipelines.
"""
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, ".")

from src.data.augmentation_auditor import audit_augmented_data
from src.data.format_repair import build_format_repair_data
from src.utils.answer_normalizer import normalize_answer_for_question


def _write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_format_repair_accepts_question_format_match():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        expr_path = root / "expr_correct.jsonl"
        repaired_path = root / "expr_format_repaired.json"
        rejected_path = root / "expr_format_rejected.json"
        report_path = root / "format_repair_report.json"
        _write_json(
            expr_path,
            [
                {
                    "id": "1",
                    "question": "甲是乙的几分之几？",
                    "answer": "3/5",
                    "expression": "120/200",
                    "eval_result": "0.6",
                    "valid": False,
                    "status": "eval_mismatch",
                },
                {
                    "id": "2",
                    "question": "结果是多少？",
                    "answer": "10",
                    "expression": "3+4",
                    "eval_result": "7",
                    "valid": False,
                    "status": "eval_mismatch",
                },
            ],
        )

        build_format_repair_data(str(expr_path), str(repaired_path), str(rejected_path), str(report_path))
        repaired = _read_json(repaired_path)
        rejected = _read_json(rejected_path)

        assert len(repaired) == 1
        assert repaired[0]["source"] == "expr_format_repair"
        assert repaired[0]["answer"] == "3/5"
        assert repaired[0]["formatted_eval_result"] == "3/5"
        assert len(rejected) == 1


def test_augmentation_audit_checks_source_replacement_not_answer_value():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / "train.json"
        expr_path = root / "expr_correct.jsonl"
        augmented_path = root / "train_augmented.json"
        clean_path = root / "train_augmented_clean.json"
        rejected_path = root / "train_augmented_rejected.json"
        report_path = root / "augmentation_report.json"
        raw_question = ['"A has 3 apples', ' and gets 2 more."']
        augmented_question = str(raw_question).replace("3", "4")

        _write_json(raw_path, [{"id": "1", "question": raw_question, "answer": "5"}])
        _write_json(expr_path, [{"id": "1", "expression": "3+2", "valid": True, "status": "ok"}])
        _write_json(
            augmented_path,
            [
                {
                    "id": "aug_1_0",
                    "source_id": "1",
                    "source": "rule_augment",
                    "question": augmented_question,
                    "answer": "6",
                    "expression": "4+2",
                    "changed_numbers": [{"old": "3", "new": "4"}],
                }
            ],
        )

        audit_augmented_data(
            str(augmented_path),
            str(raw_path),
            str(expr_path),
            str(clean_path),
            str(rejected_path),
            str(report_path),
        )
        clean = _read_json(clean_path)
        rejected = _read_json(rejected_path)
        report = _read_json(report_path)[0]

        assert len(clean) == 1
        assert not clean[0]["question"].startswith("[")
        assert clean[0]["expression"] == "4+2"
        assert clean[0]["correspondence_check"] == "source_replacement_exact"
        assert rejected == []
        assert report["stats"]["question_list_string_format"] == 1


def test_normalizer_formats_without_expression_answer_comparison():
    assert normalize_answer_for_question("0.25", "合格率是百分之几？") == "25%"
    assert normalize_answer_for_question("3.2", "至少需要几辆车？") == "4"


if __name__ == "__main__":
    test_format_repair_accepts_question_format_match()
    test_augmentation_audit_checks_source_replacement_not_answer_value()
    test_normalizer_formats_without_expression_answer_comparison()
    print("data repair pipeline tests: ALL PASSED")
