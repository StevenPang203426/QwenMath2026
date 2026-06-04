"""
Generate multiple expression candidates from an already-loaded predictor.

The module keeps model sampling details out of shell scripts so candidate
generation can be tested without loading a model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.inference.expr_predictor import expr_predict_single
from src.utils.answer_normalizer import normalize_question_text


def run_expression_candidate_inference(
    predictor,
    test_data: list[dict[str, Any]],
    source: str,
    instruction: str,
    output_csv: str,
    details_json: str,
    temperatures: list[float] | None = None,
) -> tuple[str, str]:
    temperatures = temperatures or [0.1, 0.3, 0.7]
    original_temperature = predictor.temperature
    original_do_sample = predictor.do_sample

    csv_rows: list[list[str]] = []
    details: list[dict[str, Any]] = []
    try:
        for item in tqdm(test_data, desc=source):
            item_id = str(item.get("id", ""))
            question_text = normalize_question_text(item.get("question", ""))
            first_answer = None

            for temperature in temperatures:
                predictor.temperature = temperature
                predictor.do_sample = temperature > 0.0 and temperature != min(temperatures)
                result = predictor.predict_single(question=question_text, instruction=instruction)
                pred = expr_predict_single(result.get("raw_output", ""), question_text)
                candidate_source = f"{source}_t{temperature:g}"
                details.append({
                    "id": item_id,
                    "question": question_text,
                    "source": candidate_source,
                    "temperature": temperature,
                    "expression": pred.get("expression", ""),
                    "answer": pred.get("answer", ""),
                    "eval_success": pred.get("eval_success", False),
                })
                if first_answer is None:
                    first_answer = str(pred.get("answer", "0"))

            csv_rows.append([item_id, first_answer or "0"])
    finally:
        predictor.temperature = original_temperature
        predictor.do_sample = original_do_sample

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    Path(details_json).parent.mkdir(parents=True, exist_ok=True)
    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return output_csv, details_json
