"""
Behavior tests for expression multi-candidate inference.
"""
import csv
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, ".")

from src.inference.expression_candidate_infer import run_expression_candidate_inference


class FakePredictor:
    def __init__(self):
        self.temperature = 0.1
        self.do_sample = False
        self.calls = []

    def predict_single(self, question, instruction):
        self.calls.append((question, instruction, self.temperature, self.do_sample))
        if self.temperature == 0.1:
            return {"raw_output": "<expr>16/5</expr><answer>3.2</answer>"}
        if self.temperature == 0.3:
            return {"raw_output": "<expr>3.2</expr><answer>3.2</answer>"}
        return {"raw_output": "<expr>ceil(3.2)</expr><answer>4</answer>"}


def test_expression_candidate_inference_writes_multiple_candidates_and_compat_csv():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output_csv = root / "submit_expr.csv"
        details_json = root / "expr_details.json"
        predictor = FakePredictor()

        run_expression_candidate_inference(
            predictor=predictor,
            test_data=[{"id": "1", "question": "至少需要几辆车？"}],
            source="expr_sft",
            instruction="expr instruction",
            output_csv=str(output_csv),
            details_json=str(details_json),
            temperatures=[0.1, 0.3, 0.7],
        )

        rows = list(csv.reader(output_csv.open()))
        details = json.loads(details_json.read_text())

        assert rows == [["1", "4"]]
        assert [item["expression"] for item in details] == ["16/5", "3.2", "ceil(3.2)"]
        assert [item["source"] for item in details] == ["expr_sft_t0.1", "expr_sft_t0.3", "expr_sft_t0.7"]
        assert [call[2:] for call in predictor.calls] == [(0.1, False), (0.3, True), (0.7, True)]
        assert predictor.temperature == 0.1
        assert predictor.do_sample is False


if __name__ == "__main__":
    test_expression_candidate_inference_writes_multiple_candidates_and_compat_csv()
    print("expression candidate inference tests: ALL PASSED")
