"""
Known-answer tests for the data contract.

These are the tests Team 4 runs to confirm a release is trustworthy. Each one
constructs a case with a known correct answer, so a failure means the pipeline
is wrong rather than the data being unusual.

Run: PYTHONPATH=src python3 -m pytest tests -q
"""

import numpy as np
import pandas as pd

import contract
from demo import ols_hac, risk_contributions


def _sec(n=2):
    return pd.DataFrame({
        "entity_id": ["AAA", "BBB"][:n],
        "legal_name": ["Fund A", "Fund B"][:n],
        "asset_class": ["us_large_equity", "core_bond"][:n],
        "sleeve": ["core", "bond"][:n],
        "region": ["us", "us"][:n],
        "style": ["blend", "fixed_income"][:n],
        "equity": [True, False][:n],
        "factor_model": ["ff3_mom", "not_applicable_rate_credit"][:n],
        "benchmark_proxy": ["IVV", "AGG"][:n],
        "benchmark_is_proxy": [True, True][:n],
        "liquid_designation": ["pending_client"] * n,
    })


def _periods(n, start="2015-01"):
    return [str(p) for p in pd.period_range(start, periods=n, freq="M")]


def _returns(eid="AAA", n=120, scale=1.0):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "entity_id": eid,
        "period": _periods(n),
        "total_return": rng.normal(0.006, 0.04, n) * scale,
        "source_id": "test",
        "is_proxy": False,
        "retrieved_at": "2026-09-16",
    })


def _factors(n=120, factor_set="ff3_mom"):
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "period": _periods(n),
        "factor_set": factor_set,
        "mkt_rf": rng.normal(0.006, 0.042, n),
        "smb": rng.normal(0.001, 0.026, n),
        "hml": rng.normal(0.000, 0.029, n),
        "mom": rng.normal(0.002, 0.031, n),
        "rf": np.full(n, 0.0012),
        "source_id": "test",
    })


def _cme(classes=("us_large_equity", "core_bond")):
    return pd.DataFrame({
        "asset_class": list(classes),
        "vintage": "test",
        "arithmetic_er": [0.079, 0.049],
        "compound_er": [0.067, 0.048],
        "volatility": [0.162, 0.054],
        "source_id": "test",
    })


def _corr(classes=("us_large_equity", "core_bond"), off=0.02):
    M = np.array([[1.0, off], [off, 1.0]])
    return pd.DataFrame(M, index=list(classes), columns=list(classes))


def _tables(**over):
    t = {
        "securities": _sec(),
        "returns_monthly": pd.concat([_returns("AAA"), _returns("BBB")],
                                     ignore_index=True),
        "factors_monthly": _factors(),
        "cme_inputs": _cme(),
        "cme_corr": _corr(),
    }
    t.update(over)
    return t


# --- gate tests ------------------------------------------------------------

def test_clean_release_passes():
    rep = contract.validate_release(_tables(), as_of="2024-12")
    assert rep.passed(), rep.render()


def test_duplicate_key_blocks():
    r = _returns("AAA")
    r = pd.concat([r, r.iloc[[5]]], ignore_index=True)
    rep = contract.validate_release(_tables(returns_monthly=pd.concat(
        [r, _returns("BBB")], ignore_index=True)))
    assert any(i.gate == "duplicate_key" for i in rep.blocking)


def test_percent_units_block():
    """A decimal column holding percent values must be caught, not modelled."""
    bad = _returns("AAA", scale=100.0)
    rep = contract.validate_release(_tables(returns_monthly=pd.concat(
        [bad, _returns("BBB")], ignore_index=True)))
    assert any(i.gate == "unit_scale" for i in rep.blocking)


def test_bad_period_format_blocks():
    r = _returns("AAA")
    r.loc[0, "period"] = "2015/01"
    rep = contract.validate_release(_tables(returns_monthly=r))
    assert any(i.gate == "period_format" for i in rep.blocking)


def test_orphan_entity_blocks():
    rep = contract.validate_release(_tables(returns_monthly=_returns("ZZZ")))
    assert any(i.gate == "referential" for i in rep.blocking)


def test_non_psd_correlation_blocks():
    rep = contract.validate_release(_tables(cme_corr=_corr(off=1.4)))
    gates = {i.gate for i in rep.blocking}
    assert "bounds" in gates or "psd" in gates


def test_asymmetric_correlation_blocks():
    M = _corr()
    M.iloc[0, 1] = 0.9
    rep = contract.validate_release(_tables(cme_corr=M))
    assert any(i.gate == "symmetry" for i in rep.blocking)


def test_short_history_flags_not_blocks():
    """A 40-month fund is a finding, not a pipeline error."""
    short = _returns("AAA", n=40)
    rep = contract.validate_release(_tables(returns_monthly=pd.concat(
        [short, _returns("BBB")], ignore_index=True)))
    assert rep.passed()
    assert any(i.gate == "min_observations" for i in rep.flags)


def test_calendar_gap_flagged():
    r = _returns("AAA")
    r = r[r["period"] != "2016-05"]
    rep = contract.validate_release(_tables(returns_monthly=pd.concat(
        [r, _returns("BBB")], ignore_index=True)))
    assert any(i.gate == "calendar_gap" for i in rep.flags)


def test_stale_series_flagged():
    rep = contract.validate_release(_tables(), as_of="2026-08")
    assert any(i.gate == "stale" for i in rep.flags)


# --- coverage --------------------------------------------------------------

def test_coverage_counts_gaps_and_usable_months():
    r = _returns("AAA")
    r = r[~r["period"].isin(["2016-05", "2016-06"])]
    cov = contract.build_coverage(
        pd.concat([r, _returns("BBB")], ignore_index=True), _sec(), _factors())
    row = cov[cov["entity_id"] == "AAA"].iloc[0]
    assert row["interior_gaps"] == 2
    assert row["observations"] == 118
    assert row["usable_factor_months"] == 118
    assert row["ready_for"] == "project_1_and_2"
    bond = cov[cov["entity_id"] == "BBB"].iloc[0]
    assert bond["ready_for"] == "project_1_only"


# --- model-side known answers ---------------------------------------------

def test_ols_recovers_known_coefficients():
    """Simulate with known betas, confirm the estimator finds them."""
    rng = np.random.default_rng(3)
    n = 600
    F = rng.normal(0, 0.04, (n, 4))
    true_b = np.array([0.001, 1.05, 0.30, -0.20, 0.15])
    y = true_b[0] + F @ true_b[1:] + rng.normal(0, 0.005, n)
    X = np.column_stack([np.ones(n), F])
    b, se, t, adj, rsd = ols_hac(y, X)
    assert np.allclose(b, true_b, atol=0.02)
    assert adj > 0.95


def test_risk_contributions_sum_to_volatility():
    """Euler identity. If this breaks, Project 1's attribution is wrong."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T / 100
    w = np.abs(rng.normal(size=8))
    w /= w.sum()
    sp, rc = risk_contributions(w, Sigma)
    assert np.isclose(rc.sum(), sp, rtol=1e-10)


def test_covariance_from_corr_and_vol_matches_definition():
    classes = ["us_large_equity", "core_bond"]
    cme, corr = _cme(classes), _corr(classes)
    vol = cme["volatility"].to_numpy()
    Sigma = corr.to_numpy() * np.outer(vol, vol)
    assert np.isclose(Sigma[0, 0], vol[0] ** 2)
    assert np.isclose(Sigma[0, 1], corr.iloc[0, 1] * vol[0] * vol[1])


def test_manifest_hashes_are_stable(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n")
    rep = contract.ValidationReport()
    m1 = contract.build_manifest([str(p)], "r1", "2026-08", rep)
    m2 = contract.build_manifest([str(p)], "r1", "2026-08", rep)
    assert m1["files"][0]["sha256"] == m2["files"][0]["sha256"]
    p.write_text("a,b\n1,3\n")
    m3 = contract.build_manifest([str(p)], "r1", "2026-08", rep)
    assert m3["files"][0]["sha256"] != m1["files"][0]["sha256"]
