"""
Behavior tests for expression-route training data.
"""
import json
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, ".")

from src.data.dataset import MathDataset
from src.data.expression_training_data import (
    build_expression_clean_splits,
    build_expression_dpo_pairs,
)
from src.utils.config import load_config


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self._id_to_char = {0: "<pad>"}

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(ch) for ch in text]
        for ch in text:
            self._id_to_char[ord(ch)] = ch
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        return "".join(self._id_to_char.get(i, "") for i in ids if i != self.pad_token_id)


def _write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decoded_label(dataset: MathDataset) -> str:
    item = dataset[0]
    label_ids = [token for token in item["labels"] if token != -100 and token != dataset.tokenizer.pad_token_id]
    return dataset.tokenizer.decode(label_ids)


def test_expression_clean_splits_keep_only_safe_expression_records():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clean_path = root / "train_clean_augmented.json"
        output_path = root / "train_expr_clean.json"
        train_path = root / "train_expr_clean_train.json"
        val_path = root / "train_expr_clean_val.json"
        report_path = root / "train_expr_clean_report.json"

        _write_json(clean_path, [
            {"id": "1", "source_id": "1", "source": "raw_ok", "question": "至少需要几辆车？", "answer": "4", "expression": "16/5"},
            {"id": "2", "source_id": "2", "source": "raw_ok", "question": "结果是多少？", "answer": "1", "expression": "5%2"},
            {"id": "3", "source_id": "3", "source": "auto_repair", "question": "结果是多少？", "answer": "9", "expression": "4+5"},
            {"id": "4", "source_id": "4", "source": "rule_augment", "question": "结果是多少？", "answer": "8", "expression": "3+4"},
        ])

        records = build_expression_clean_splits(
            input_path=str(clean_path),
            output_path=str(output_path),
            train_output=str(train_path),
            val_output=str(val_path),
            report_output=str(report_path),
            val_ratio=0.5,
            seed=7,
        )

        assert [item["id"] for item in records] == ["1", "3"]
        assert all("<expr>" not in item["answer"] for item in records)
        assert len(_read_json(train_path)) + len(_read_json(val_path)) == 2
        report = _read_json(report_path)[0]
        assert report["kept"] == 2
        assert report["rejected"] == 2
        assert report["reject_reasons"]["policy_violation"] == 1
        assert report["reject_reasons"]["answer_mismatch"] == 1


def test_expression_dpo_pairs_use_parseable_wrong_expressions_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        positives_path = root / "train_expr_clean.json"
        wrong_path = root / "expr_wrong.json"
        output_path = root / "train_expr_dpo_clean.json"
        report_path = root / "train_expr_dpo_clean_report.json"

        _write_json(positives_path, [
            {"id": "1", "source_id": "1", "source": "raw_ok", "question": "结果是多少？", "answer": "5", "expression": "2+3"},
            {"id": "aug_1", "source_id": "1", "source": "rule_augment", "question": "增强题", "answer": "7", "expression": "3+4"},
        ])
        _write_json(wrong_path, [
            {"id": "1", "question": "结果是多少？", "answer": "5", "expression": "2+2", "status": "ok", "valid": False, "eval_result": "4"},
            {"id": "aug_1", "question": "增强题", "answer": "7", "expression": "3+4", "status": "accidentally_correct", "valid": True, "eval_result": "7"},
        ])

        pairs = build_expression_dpo_pairs(
            positive_path=str(positives_path),
            wrong_path=str(wrong_path),
            output_path=str(output_path),
            report_output=str(report_path),
        )

        assert len(pairs) == 1
        assert pairs[0]["id"] == "1"
        assert pairs[0]["chosen"] == "<expr>2+3</expr><answer>5</answer>"
        assert pairs[0]["rejected"] == "<expr>2+2</expr><answer>4</answer>"
        assert _read_json(report_path)[0]["paired"] == 1


def test_math_dataset_can_train_expression_targets():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_expr_clean.json"
        _write_json(path, [{
            "id": "1",
            "question": "结果是多少？",
            "answer": "5",
            "expression": "2+3",
            "instruction": "expr instruction",
        }])

        dataset = MathDataset(
            data_path=str(path),
            tokenizer=TinyTokenizer(),
            max_length=512,
            target_format="expression",
        )

        assert _decoded_label(dataset) == "<expr>2+3</expr><answer>5</answer>"


def test_expression_clean_configs_select_expression_targets_and_stable_grpo():
    sft = load_config("configs/sft_expr_clean.yaml")
    grpo = load_config("configs/grpo_expr_clean.yaml")
    grpo_from_dpo = load_config("configs/grpo_expr_from_dpo_clean.yaml")

    assert sft.data.target_format == "expression"
    assert sft.data.train_path == "data/splits/train_expr_clean_train.json"
    assert grpo.grpo.use_vllm is False
    assert grpo_from_dpo.grpo.use_vllm is False


if __name__ == "__main__":
    test_expression_clean_splits_keep_only_safe_expression_records()
    test_expression_dpo_pairs_use_parseable_wrong_expressions_only()
    test_math_dataset_can_train_expression_targets()
    test_expression_clean_configs_select_expression_targets_and_stable_grpo()
    print("expression training data tests: ALL PASSED")
