"""
Behavior tests for expression-first CSV/JSON fusion.
"""
import csv
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, ".")

from src.inference.expression_vote_csv import run_expression_first_vote


def test_expression_vote_csv_uses_expression_details_as_base():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_path = root / "test.json"
        expr_path = root / "expr.json"
        cot_path = root / "cot.csv"
        output_path = root / "submit.csv"

        test_path.write_text(json.dumps([
            {"id": "1", "question": "至少需要几辆车？"},
            {"id": "2", "question": "结果是多少？"},
        ], ensure_ascii=False))
        expr_path.write_text(json.dumps([
            {"id": "1", "expression": "16/5"},
            {"id": "1", "expression": "ceil(3.2)"},
            {"id": "2", "expression": "x+1=3"},
        ], ensure_ascii=False))
        with cot_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["1", "3"])
            writer.writerow(["2", "12"])

        run_expression_first_vote(
            test_path=str(test_path),
            output_path=str(output_path),
            expr_specs=[f"{expr_path}:expr:1.5"],
            cot_specs=[f"{cot_path}:cot:1.0"],
        )

        rows = list(csv.reader(output_path.open()))
        assert rows == [["1", "4"], ["2", "12"]]


if __name__ == "__main__":
    test_expression_vote_csv_uses_expression_details_as_base()
    print("expression vote csv tests: ALL PASSED")
