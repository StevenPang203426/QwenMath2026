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
from src.data.merge_clean_data import merge_clean_data
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


def test_clean_data_merge_reports_source_composition():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / "train.json"
        audit_path = root / "quality_audit.json"
        expr_path = root / "expr_correct.json"
        repaired_path = root / "train_repaired.json"
        format_repaired_path = root / "expr_format_repaired.json"
        augmented_path = root / "train_augmented_clean.json"
        output_path = root / "train_clean_augmented.json"
        unified_path = root / "train_repairs_unified.json"
        report_path = root / "train_clean_augmented_report.json"

        _write_json(raw_path, [
            {"id": "1", "question": "原始合格题", "answer": "2"},
            {"id": "2", "question": "题意待修题", "answer": "3"},
            {"id": "3", "question": "格式待修题", "answer": "4"},
        ])
        _write_json(audit_path, [])
        _write_json(expr_path, [{"id": "1", "expression": "1+1", "valid": True, "status": "ok"}])
        _write_json(repaired_path, [{
            "id": "auto_2",
            "source_id": "2",
            "source": "auto_repair",
            "question": "题意修复题",
            "answer": "3",
            "expression": "1+2",
        }])
        _write_json(format_repaired_path, [{
            "id": "format_repair_3",
            "source_id": "3",
            "source": "expr_format_repair",
            "question": "表达式格式修复题",
            "answer": "4",
            "expression": "16/5",
        }])
        _write_json(augmented_path, [{
            "id": "aug_1_0",
            "source_id": "1",
            "source": "rule_augment",
            "question": "增强题",
            "answer": "5",
            "expression": "2+3",
        }])

        merge_clean_data(
            raw_path=str(raw_path),
            audit_path=str(audit_path),
            expr_path=str(expr_path),
            repaired_path=str(repaired_path),
            format_repaired_path=str(format_repaired_path),
            augmented_path=str(augmented_path),
            output_path=str(output_path),
            unified_repairs_output=str(unified_path),
            report_output=str(report_path),
        )

        report = _read_json(report_path)[0]
        assert report["total_records"] == 4
        assert report["source_counts"] == {
            "raw_ok": 1,
            "auto_repair": 1,
            "expr_format_repair": 1,
            "rule_augment": 1,
        }
        assert report["source_percentages"] == {
            "raw_ok": 25.0,
            "auto_repair": 25.0,
            "expr_format_repair": 25.0,
            "rule_augment": 25.0,
        }


if __name__ == "__main__":
    test_format_repair_accepts_question_format_match()
    test_augmentation_audit_checks_source_replacement_not_answer_value()
    test_normalizer_formats_without_expression_answer_comparison()
    test_clean_data_merge_reports_source_composition()
    print("data repair pipeline tests: ALL PASSED")
