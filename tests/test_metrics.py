"""
Behavior tests for metric numeric normalization.
"""
import sys

sys.path.insert(0, ".")

from src.utils.metrics import normalize_number


def test_normalize_number_rejects_nonfinite_float_results():
    huge = "9" * 400

    assert normalize_number(huge) is None
    assert normalize_number(f"{huge}%") is None
    assert normalize_number(f"{huge}/1") is None
    assert normalize_number("24") == "24"


if __name__ == "__main__":
    test_normalize_number_rejects_nonfinite_float_results()
    print("metrics tests: ALL PASSED")
