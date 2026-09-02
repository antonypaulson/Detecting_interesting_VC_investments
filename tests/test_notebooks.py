"""Sanity checks that the analysis notebooks still parse."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "TCG_scoring_model.ipynb",
    "TCG_Data_preparation.ipynb",
    "EDA_and_Machine_Learning_TCG.ipynb",
]


def test_notebooks_are_valid_nbformat():
    for name in NOTEBOOKS:
        nb = json.loads((ROOT / name).read_text())
        assert nb["nbformat"] == 4
        assert any(
            "vc_investments" in "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell["cell_type"] == "code"
        ), name
