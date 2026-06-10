#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"

"${PYTHON_BIN}" -m src.inference.expr4_eval_submit submit "$@"
