"""Hashing, manifests, lineage and configuration."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from quantproof.config import AuditConfig, ValidationConfig
from quantproof.data import generate_prices
from quantproof.errors import QuantProofConfigError
from quantproof.experiments import (
    Lineage,
    build_manifest,
    canonical_json,
    hash_config,
    hash_dataframe,
    hash_file,
    manifest_to_yaml,
    schema_of,
)


def test_dataframe_hash_deterministic_and_sensitive():
    a = generate_prices(100, seed=1)
    assert hash_dataframe(a) == hash_dataframe(a.copy())
    b = a.copy()
    b.iloc[50, 0] = np.nextafter(b.iloc[50, 0], np.inf)
    assert hash_dataframe(a) != hash_dataframe(b)
    assert hash_dataframe(a) != hash_dataframe(a.iloc[::-1])
    assert hash_dataframe(a) != hash_dataframe(a.rename(columns={"open": "o"}))
    assert hash_dataframe(a) != hash_dataframe(a.astype({"volume": float}))


def test_hash_independent_of_datetime_resolution_and_timezone_representation():
    a = generate_prices(10, seed=2)
    b = a.copy()
    b.index = b.index.as_unit("s")
    assert hash_dataframe(a) == hash_dataframe(b)
    utc = a.tz_localize("UTC")
    ny = utc.tz_convert("America/New_York")
    assert hash_dataframe(utc) == hash_dataframe(ny)  # same instants


def test_hash_canonicalizes_nan_payloads():
    x = pd.DataFrame({"v": [1.0, np.nan]})
    weird = np.array([1.0, 0.0])
    weird.view(np.uint64)[1] = 0x7FF8000000000123  # NaN with a payload
    y = pd.DataFrame({"v": weird})
    assert hash_dataframe(x) == hash_dataframe(y)


def test_hash_known_object_columns():
    df = pd.DataFrame({"s": ["a", None, "b"], "i": [1, 2, 3]})
    assert len(hash_dataframe(df)) == 64
    assert hash_dataframe(df) == hash_dataframe(df.copy())


def test_file_and_config_hash(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    assert hash_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_config({"b": 1, "a": [1, 2]}) == hash_config({"a": [1, 2], "b": 1})
    assert canonical_json({"x": float("nan")}) == '{"x":null}'


def test_manifest_content_hash_ignores_timestamp():
    kwargs = {
        "strategy": {"name": "s"},
        "data": {"sha256": "x"},
        "config": {"a": 1},
        "validation": {},
        "execution": {},
        "seed": 1,
    }
    m1 = build_manifest(**kwargs, now=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc))
    m2 = build_manifest(**kwargs, now=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc))
    assert m1["content_hash"] == m2["content_hash"] and m1["created_at"] != m2["created_at"]
    m3 = build_manifest(**{**kwargs, "seed": 2})
    assert m3["content_hash"] != m1["content_hash"]
    assert "quantproof_version" in manifest_to_yaml(m1)
    assert m1["environment"]["packages"]["numpy"]


def test_schema_and_lineage():
    df = generate_prices(5, seed=1)
    s = schema_of(df)
    assert s["rows"] == 5 and s["columns"]["close"] == "float64" and s["start"].startswith("2015")
    lin = Lineage()
    lin.record("load", inputs={"raw": "a"}, outputs={"clean": "b"}, notes=["sorted"])
    assert lin.to_list()[0]["step"] == "load"


def test_config_file_formats(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "seed: 7\nexecution:\n  commission_bps: 2\n  signal_lag: 2\nstatistics:\n  trials: 500\n"
    )
    (tmp_path / "c.json").write_text(json.dumps({"validation": {"embargo": 0.02}}))
    (tmp_path / "c.toml").write_text("[statistics]\nannualization_check = 1\n")
    cfg = AuditConfig.from_file(tmp_path / "c.yaml")
    assert cfg.seed == 7 and cfg.execution.signal_lag == 2 and cfg.statistics.trials == 500
    assert AuditConfig.from_file(tmp_path / "c.json").validation.embargo == 0.02
    with pytest.raises(QuantProofConfigError, match=r"statistics\.annualization_check"):
        AuditConfig.from_file(tmp_path / "c.toml")
    with pytest.raises(QuantProofConfigError, match="not found"):
        AuditConfig.from_file(tmp_path / "missing.yaml")
    (tmp_path / "c.ini").write_text("x")
    with pytest.raises(QuantProofConfigError, match="Unsupported"):
        AuditConfig.from_file(tmp_path / "c.ini")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(QuantProofConfigError, match="mapping"):
        AuditConfig.from_file(tmp_path / "list.yaml")


def test_config_rejects_typos_and_bad_values():
    with pytest.raises(QuantProofConfigError, match="comission_bps"):
        AuditConfig.from_dict({"execution": {"comission_bps": 2}})
    with pytest.raises(ValueError, match="even"):
        ValidationConfig(pbo_partitions=7)
    with pytest.raises(ValueError, match="smaller than cpcv_groups"):
        ValidationConfig(cpcv_groups=4, cpcv_test_groups=4)


def test_quick_mode_is_explicit_and_copy():
    cfg = AuditConfig(quick=True)
    eff = cfg.effective()
    assert eff.statistics.n_bootstrap == 200 and eff.causality.n_timestamps == 4
    assert cfg.statistics.n_bootstrap == 1000  # original untouched
    assert AuditConfig().effective().statistics.n_bootstrap == 1000
    assert AuditConfig().execution.one_way_cost_bps == pytest.approx(1 + 1 + 2)
