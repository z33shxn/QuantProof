"""SVG chart helpers and serialization utilities."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import math
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantproof._utils import as_float_array, format_number, to_jsonable
from quantproof.errors import QuantProofInputError
from quantproof.reports.charts import bar_chart, heatmap, histogram, line_chart


def test_line_chart_escapes_and_embeds_data():
    svg = line_chart(
        [
            {
                "name": "a<b>",
                "short": "A",
                "x": ["2024-01-01", "2024-01-02"],
                "y": [1.0, 1.1],
                "slot": 1,
            },
            {"name": "b", "x": ["2024-01-01", "2024-01-02"], "y": [1.0, None], "slot": 2},
        ],
        chart_id="c1",
    )
    assert "a&lt;b&gt;" in svg and '<script type="application/json"' in svg
    payload = svg.split('data-for="c1">')[1].split("</script>")[0]
    assert json.loads(payload)["series"][1]["y"] == [1.0, None]
    assert line_chart([{"name": "x", "x": ["a"], "y": [None]}]) == ""
    flat = line_chart([{"name": "flat", "x": ["a", "b"], "y": [1.0, 1.0]}])
    assert "<path" in flat


def test_bar_chart_negative_values_and_highlight():
    svg = bar_chart(["0", "1", "2"], [0.5, -0.4, None], highlight=0)
    assert svg.count("<rect") == 2 and "neg" in svg and " hl" in svg
    assert bar_chart(["a"], [None]) == ""
    assert "<rect" in bar_chart(["a", "b"], [0.0, 0.0])


def test_histogram_and_reference_line():
    svg = histogram(list(np.linspace(-2, 3, 50)), bins=10, vline=0.0)
    assert 'class="refline"' in svg and svg.count('class="bar s1"') == 10
    assert histogram([1.0]) == ""
    assert "<rect" in histogram([1.0, 1.0, 1.0], vline=None)


def test_heatmap_diverging_and_best_marker():
    svg = heatmap([1, 2], [10, 20], [[1.0, -0.5], [None, 0.0]], x_label="fast", y_label="slow")
    assert "★" in svg and "var(--div-pos-5)" in svg and "var(--div-neg-3)" in svg
    assert 'class="cell empty"' in svg and "var(--div-mid)" in svg
    assert heatmap([1], [1], [[None]], x_label="a", y_label="b") == ""


class Color(Enum):
    RED = "red"


def test_to_jsonable_covers_types():
    obj = {
        "np_int": np.int64(3),
        "np_float": np.float32(1.5),
        "nan": float("nan"),
        "inf": np.inf,
        "bool": np.bool_(True),
        "ts": pd.Timestamp("2024-01-01"),
        "date": dt.date(2024, 1, 2),
        "td": pd.Timedelta(days=1),
        "path": Path("a/b"),
        "enum": Color.RED,
        "series": pd.Series([1, 2], index=["a", "b"]),
        "frame": pd.DataFrame({"x": [1]}),
        "arr": np.array([1.0, np.nan]),
        "tuple": (1, 2),
        "set": {3},
        "other": object,
    }
    out = to_jsonable(obj)
    json.dumps(out, allow_nan=False)
    assert out["nan"] is None and out["inf"] is None and out["enum"] == "red"
    assert out["series"] == {"a": 1, "b": 2} and out["frame"] == [{"x": 1}]


def test_small_utils():
    assert format_number(None) == "n/a" and format_number(float("nan")) == "n/a"
    assert format_number("x") == "x" and format_number(1.23456, 2) == "1.23"
    assert as_float_array(5.0).shape == (1,)
    with pytest.raises(QuantProofInputError):
        as_float_array(np.zeros((2, 2)))
    from quantproof._utils import clean_returns

    with pytest.raises(QuantProofInputError):
        clean_returns([1.0, np.nan], nan_policy="bogus")


def test_lazy_audit_exports():
    audit_pkg = importlib.import_module("quantproof.audit")
    assert callable(audit_pkg.audit) and audit_pkg.LIMITATIONS
    with pytest.raises(AttributeError):
        _ = audit_pkg.not_a_thing
    assert math.isfinite(1.0)
