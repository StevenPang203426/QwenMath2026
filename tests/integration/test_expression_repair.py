"""
Behavior tests for the idempotent expression repair pipeline.
"""
import json
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, ".")

from src.data.expression_repair import build_repair_messages, repair_expression_records


def _write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_expression_repair_is_idempotent_and_writes_only_safe_data():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / "train.json"
        expr_path = root / "expr_correct.json"
        safe_path = root / "expr_safe.json"
        repaired_path = root / "expr_repaired.json"
        rejected_path = root / "expr_rejected.json"
        report_path = root / "expr_report.json"
        sft_path = root / "train_expr_safe.json"

        _write_json(
            raw_path,
            [
                {"id": "1", "question": "3盒彩笔，每盒4支，一共多少支？", "answer": "12"},
                {"id": "2", "question": "至少需要几辆车？", "answer": "4"},
                {"id": "3", "question": "5减2是多少？", "answer": "3"},
            ],
        )
        _write_json(
            expr_path,
            [
                {"id": "1", "question": "3盒彩笔，每盒4支，一共多少支？", "answer": "12", "expression": "3*4", "valid": True, "status": "ok"},
                {"id": "2", "question": "至少需要几辆车？", "answer": "4", "expression": "ceil(3.2)", "valid": False, "status": "eval_mismatch"},
                {"id": "3", "question": "5减2是多少？", "answer": "3", "expression": "5>2", "valid": False, "status": "eval_mismatch"},
            ],
        )

        calls = []

        def fake_request(item, attempt, messages):
            calls.append((item["id"], attempt, messages))
            if item["id"] == "2":
                return {"expression": "16/5", "raw_response": "<expr>16/5</expr>"}
            return {"expression": "5-1", "raw_response": "<expr>5-1</expr>"}

        repair_expression_records(
            raw_path=str(raw_path),
            expr_path=str(expr_path),
            safe_output=str(safe_path),
            repaired_output=str(repaired_path),
            rejected_output=str(rejected_path),
            report_output=str(report_path),
            sft_output=str(sft_path),
            request_fn=fake_request,
            max_attempts=2,
        )

        assert [call[0] for call in calls] == ["2", "3", "3"]
        user_text = calls[0][2][1]["content"][0]["text"]
        assert user_text.index("【规范表达式要求】") < user_text.index("【候选题目 JSON】")
        assert "ceil(3.2)" in user_text

        safe = _read_json(safe_path)
        repaired = _read_json(repaired_path)
        rejected = _read_json(rejected_path)
        report = _read_json(report_path)[0]
        sft = _read_json(sft_path)

        assert [item["id"] for item in safe] == ["1", "2"]
        assert repaired[0]["id"] == "2"
        assert repaired[0]["expression"] == "16/5"
        assert repaired[0]["eval_result"] == "4"
        assert [item["id"] for item in sft] == ["1", "2"]
        assert sft[1]["expression"] == "16/5"
        assert rejected[0]["id"] == "3"
        assert rejected[0]["reject_reason"] == "answer_mismatch"
        assert report["stats"]["safe_existing"] == 1
        assert report["stats"]["repaired"] == 1
        assert report["stats"]["rejected"] == 1


def test_expression_repair_can_write_safe_artifacts_without_api_calls():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / "train.json"
        expr_path = root / "expr_correct.json"
        safe_path = root / "expr_safe.json"
        repaired_path = root / "expr_repaired.json"
        rejected_path = root / "expr_rejected.json"
        report_path = root / "expr_report.json"
        sft_path = root / "train_expr_safe.json"

        _write_json(raw_path, [
            {"id": "1", "question": "3盒彩笔，每盒4支，一共多少支？", "answer": "12"},
            {"id": "2", "question": "结果是多少？", "answer": "1"},
        ])
        _write_json(expr_path, [
            {"id": "1", "question": "3盒彩笔，每盒4支，一共多少支？", "answer": "12", "expression": "3*4", "valid": True, "status": "ok"},
            {"id": "2", "question": "结果是多少？", "answer": "1", "expression": "5%2", "valid": True, "status": "ok"},
        ])

        repair_expression_records(
            raw_path=str(raw_path),
            expr_path=str(expr_path),
            safe_output=str(safe_path),
            repaired_output=str(repaired_path),
            rejected_output=str(rejected_path),
            report_output=str(report_path),
            sft_output=str(sft_path),
            max_attempts=0,
        )

        assert [item["id"] for item in _read_json(safe_path)] == ["1"]
        assert _read_json(repaired_path) == []
        rejected = _read_json(rejected_path)
        assert rejected[0]["id"] == "2"
        assert rejected[0]["reject_reason"] == "policy_violation"
        assert _read_json(report_path)[0]["stats"]["rejected"] == 1


def test_expression_repair_checkpoints_partial_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / "train.json"
        expr_path = root / "expr_correct.json"
        safe_path = root / "expr_safe.json"
        repaired_path = root / "expr_repaired.json"
        rejected_path = root / "expr_rejected.json"
        report_path = root / "expr_report.json"
        sft_path = root / "train_expr_safe.json"

        _write_json(raw_path, [
            {"id": "1", "question": "3盒彩笔，每盒4支，一共多少支？", "answer": "12"},
            {"id": "2", "question": "5减2是多少？", "answer": "3"},
        ])
        _write_json(expr_path, [
            {"id": "1", "question": "3盒彩笔，每盒4支，一共多少支？", "answer": "12", "expression": "3*4", "valid": True, "status": "ok"},
            {"id": "2", "question": "5减2是多少？", "answer": "3", "expression": "5>2", "valid": False, "status": "eval_mismatch"},
        ])

        def fake_request(item, attempt, messages):
            assert item["id"] == "2"
            assert [row["id"] for row in _read_json(safe_path)] == ["1"]
            assert _read_json(report_path)[0]["complete"] is False
            return {"expression": "5-2", "raw_response": "<expr>5-2</expr>"}

        repair_expression_records(
            raw_path=str(raw_path),
            expr_path=str(expr_path),
            safe_output=str(safe_path),
            repaired_output=str(repaired_path),
            rejected_output=str(rejected_path),
            report_output=str(report_path),
            sft_output=str(sft_path),
            request_fn=fake_request,
            max_attempts=1,
            checkpoint_every=1,
        )

        report = _read_json(report_path)[0]
        assert report["complete"] is True
        assert report["processed"] == 2
        assert [item["id"] for item in _read_json(sft_path)] == ["1", "2"]


if __name__ == "__main__":
    test_expression_repair_is_idempotent_and_writes_only_safe_data()
    test_expression_repair_can_write_safe_artifacts_without_api_calls()
    test_expression_repair_checkpoints_partial_outputs()
    print("expression repair tests: ALL PASSED")
