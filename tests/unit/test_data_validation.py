"""Data-quality rules QP-DATA-001..013 and loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof.config import DataValidationConfig
from quantproof.data import (
    generate_prices,
    load_frame,
    load_prices,
    load_returns,
    prepare_prices,
    validate_data,
)
from quantproof.data.loaders import parse_timestamps
from quantproof.errors import QuantProofDataError
from quantproof.severity import Severity


def by_id(findings):
    return {f.id: f for f in findings}


def test_clean_synthetic_data_passes_everything():
    fs = validate_data(generate_prices(300, seed=1))
    assert {f.severity for f in fs} == {Severity.PASS}
    assert len(fs) == 13


def raw_frame(n=40):
    df = generate_prices(n, seed=2).reset_index()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def test_invalid_timestamps():
    df = raw_frame()
    df.loc[3, "timestamp"] = "not a date"
    f = by_id(validate_data(df))["QP-DATA-001"]
    assert f.severity is Severity.WARN and f.evidence["count"] == 1
    assert "not a date" in f.evidence["examples"]


def test_non_monotonic_message_names_timestamps():
    df = raw_frame()
    df.loc[[5, 6], "timestamp"] = df.loc[[6, 5], "timestamp"].to_numpy()
    f = by_id(validate_data(df))["QP-DATA-002"]
    assert f.severity is Severity.WARN
    assert "First problematic timestamp" in f.message and "previous timestamp" in f.message


def test_duplicate_timestamps_and_rows():
    df = raw_frame()
    dup = pd.concat([df, df.iloc[[10, 11]]], ignore_index=True)
    fs = by_id(validate_data(dup))
    assert fs["QP-DATA-004"].severity is Severity.WARN and fs["QP-DATA-004"].evidence["count"] == 2
    assert fs["QP-DATA-005"].severity is Severity.WARN
    assert fs["QP-DATA-004"].location.start is not None
    allowed = by_id(validate_data(dup, DataValidationConfig(allow_duplicate_timestamps=True)))
    assert allowed["QP-DATA-004"].severity is Severity.INFO


def test_missing_ohlc_and_required():
    df = generate_prices(30, seed=3)[["close"]]
    assert by_id(validate_data(df))["QP-DATA-003"].severity is Severity.INFO
    strict = DataValidationConfig(require_ohlc=True)
    assert by_id(validate_data(df, strict))["QP-DATA-003"].severity is Severity.FAIL
    no_price = generate_prices(30, seed=3)[["volume"]]
    assert by_id(validate_data(no_price))["QP-DATA-003"].severity is Severity.FAIL


def test_impossible_ohlc():
    df = generate_prices(30, seed=4)
    df.iloc[7, df.columns.get_loc("high")] = df["low"].iloc[7] * 0.5
    assert by_id(validate_data(df))["QP-DATA-006"].severity is Severity.FAIL


def test_non_positive_prices_and_infinite():
    df = generate_prices(30, seed=5)
    df.iloc[4, df.columns.get_loc("close")] = 0.0
    df["volume"] = df["volume"].astype(float)
    df.iloc[6, df.columns.get_loc("volume")] = np.inf
    fs = by_id(validate_data(df))
    assert fs["QP-DATA-007"].severity is Severity.FAIL
    assert fs["QP-DATA-012"].severity is Severity.FAIL


def test_nulls_and_stale_prices():
    df = generate_prices(60, seed=6)
    df.iloc[10, df.columns.get_loc("open")] = np.nan
    df.iloc[20:28, df.columns.get_loc("close")] = df["close"].iloc[20]
    fs = by_id(validate_data(df))
    assert fs["QP-DATA-008"].evidence["by_column"] == {"open": 1}
    assert (
        fs["QP-DATA-011"].severity is Severity.WARN and fs["QP-DATA-011"].evidence["longest"] == 8
    )


def test_gaps_daily_and_intraday():
    df = generate_prices(100, seed=7)
    gappy = df.drop(df.index[40:52])
    f = by_id(validate_data(gappy))["QP-DATA-009"]
    assert f.severity is Severity.WARN
    assert f.evidence["median_spacing"] == "1 days 00:00:00"
    # weekends (3-day gaps) are not flagged at the default 5x threshold
    assert by_id(validate_data(df))["QP-DATA-009"].severity is Severity.PASS
    idx = pd.date_range("2024-01-02 09:30", periods=30, freq="min").append(
        pd.date_range("2024-01-03 09:30", periods=30, freq="min")
    )
    intraday = pd.DataFrame({"close": np.linspace(100, 101, 60)}, index=idx)
    f2 = by_id(validate_data(intraday))["QP-DATA-009"]
    assert f2.severity is Severity.PASS and f2.evidence["session_breaks_ignored"] == 1


def test_timezone_mixtures():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-03-08 10:00-05:00", "2024-03-11 10:00-04:00", "2024-03-12 10:00"],
            "close": [1, 2, 3],
        }
    )
    assert by_id(validate_data(df))["QP-DATA-010"].severity is Severity.WARN
    dst = df.iloc[:2]
    assert by_id(validate_data(dst))["QP-DATA-010"].severity is Severity.INFO


def test_extreme_returns():
    df = generate_prices(50, seed=8)
    df.iloc[25:, df.columns.get_loc("close")] *= 3
    f = by_id(validate_data(df))["QP-DATA-013"]
    assert f.severity is Severity.WARN and f.evidence["largest"] == pytest.approx(2.0, rel=0.1)


def test_disabled_rules_and_disabled_validation():
    df = generate_prices(30, seed=9)
    fs = validate_data(df, DataValidationConfig(disabled_rules=["QP-DATA-013"]))
    assert "QP-DATA-013" not in {f.id for f in fs}
    assert validate_data(df, DataValidationConfig(enabled=False)) == []


def test_prepare_prices_records_actions():
    df = raw_frame(20)
    df = pd.concat([df.iloc[::-1], df.iloc[[0]]], ignore_index=True)
    prepared = prepare_prices(df)
    assert prepared.prices.index.is_monotonic_increasing
    assert not prepared.prices.index.duplicated().any()
    assert any("Sorted" in a for a in prepared.actions)
    assert any("duplicated" in a for a in prepared.actions)


def test_loader_errors_are_actionable(tmp_path):
    with pytest.raises(QuantProofDataError, match="not found"):
        load_frame(tmp_path / "missing.csv")
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(QuantProofDataError, match="Unsupported"):
        load_frame(tmp_path / "x.txt")
    with pytest.raises(QuantProofDataError, match="timestamp column"):
        prepare_prices(pd.DataFrame({"close": [1.0, 2.0]}))
    with pytest.raises(QuantProofDataError, match="epoch"):
        parse_timestamps(pd.Series([1, 2, 3]))
    with pytest.raises(QuantProofDataError, match="Timestamp column"):
        load_frame(pd.DataFrame({"a": [1]}), timestamp_column="ts")


def test_csv_and_parquet_round_trip(tmp_path):
    df = generate_prices(20, seed=10)
    df.to_csv(tmp_path / "p.csv")
    df.to_parquet(tmp_path / "p.parquet")
    a = load_prices(tmp_path / "p.csv")
    b = load_prices(tmp_path / "p.parquet")
    assert np.allclose(a["close"], b["close"]) and len(a) == 20
    pd.DataFrame({"date": df.index, "returns": df["close"].pct_change()}).to_csv(
        tmp_path / "r.csv", index=False
    )
    r = load_returns(tmp_path / "r.csv")
    assert r.name == "returns" and len(r) == 20


def test_column_aliases_normalized():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Adj Close": [1.0, 1.1]})
    frame = load_frame(df)
    assert list(frame.columns) == ["date", "adj_close"]
