"""
Data contract for the First Bank practicum.

Defines the four release tables, the validation gates that must pass before a
release is tagged, and the coverage table that tells Teams 2 and 3 what they can
actually model. Written with pandas and numpy only so it runs anywhere the team
has a plain Python install.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIOD_RE = r"^\d{4}-(0[1-9]|1[0-2])$"

# ---------------------------------------------------------------------------
# Table schemas. Column order is part of the contract.
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict] = {
    "securities": {
        "key": ["entity_id"],
        "columns": {
            "entity_id": "str",
            "legal_name": "str",
            "asset_class": "str",
            "sleeve": "str",
            "region": "str",
            "style": "str",
            "equity": "bool",
            "factor_model": "str",
            "benchmark_proxy": "str",
            "benchmark_is_proxy": "bool",
            "liquid_designation": "str",
        },
    },
    "returns_monthly": {
        "key": ["entity_id", "period"],
        "columns": {
            "entity_id": "str",
            "period": "period",
            "total_return": "float",
            "source_id": "str",
            "is_proxy": "bool",
            "retrieved_at": "str",
        },
    },
    "factors_monthly": {
        "key": ["period", "factor_set"],
        "columns": {
            "period": "period",
            "factor_set": "str",
            "mkt_rf": "float",
            "smb": "float",
            "hml": "float",
            "mom": "float",
            "rf": "float",
            "source_id": "str",
        },
    },
    "cme_inputs": {
        "key": ["asset_class"],
        "columns": {
            "asset_class": "str",
            "vintage": "str",
            "arithmetic_er": "float",
            "compound_er": "float",
            "volatility": "float",
            "source_id": "str",
        },
    },
    "policy_weights": {
        "key": ["entity_id"],
        "columns": {
            "entity_id": "str",
            "policy_weight": "float",
            "asset_class": "str",
            "weight_type": "str",
            "as_of": "str",
            "source_id": "str",
        },
    },
}

# Plausibility bands used by the unit-error gate. A monthly total return outside
# these bounds is almost always a percent value sitting in a decimal column.
RETURN_ABS_WARN = 0.60
RETURN_ABS_SHARE_LIMIT = 0.02
CME_ER_ABS_WARN = 0.40
CME_VOL_BOUNDS = (0.005, 0.60)


@dataclass
class Issue:
    table: str
    gate: str
    severity: str  # "block" or "flag"
    detail: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, table, gate, severity, detail):
        self.issues.append(Issue(table, gate, severity, detail))

    @property
    def blocking(self):
        return [i for i in self.issues if i.severity == "block"]

    @property
    def flags(self):
        return [i for i in self.issues if i.severity == "flag"]

    def passed(self):
        return len(self.blocking) == 0

    def to_frame(self):
        if not self.issues:
            return pd.DataFrame(columns=["table", "gate", "severity", "detail"])
        return pd.DataFrame([i.__dict__ for i in self.issues])

    def render(self):
        lines = []
        status = "PASS" if self.passed() else "BLOCKED"
        lines.append(f"Validation status: {status}")
        lines.append(f"  blocking issues: {len(self.blocking)}")
        lines.append(f"  flags:           {len(self.flags)}")
        for i in self.issues:
            mark = "BLOCK" if i.severity == "block" else " flag"
            lines.append(f"  [{mark}] {i.table}.{i.gate}: {i.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _check_schema(df, name, report):
    spec = SCHEMAS[name]
    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        report.add(name, "schema", "block", f"missing columns {missing}")
    extra = [c for c in df.columns if c not in spec["columns"]]
    if extra:
        report.add(name, "schema", "flag", f"undeclared columns {extra}")

    for col, kind in spec["columns"].items():
        if col not in df.columns:
            continue
        if kind == "period":
            bad = df[~df[col].astype(str).str.match(PERIOD_RE, na=False)]
            if len(bad):
                report.add(name, "period_format", "block",
                           f"{len(bad)} rows where {col} is not YYYY-MM")
        elif kind == "float":
            if not pd.api.types.is_numeric_dtype(df[col]):
                report.add(name, "dtype", "block", f"{col} is not numeric")


def _check_keys(df, name, report):
    key = SCHEMAS[name]["key"]
    if any(k not in df.columns for k in key):
        return
    dup = df[df.duplicated(subset=key, keep=False)]
    if len(dup):
        sample = dup[key].drop_duplicates().head(3).to_dict("records")
        report.add(name, "duplicate_key", "block",
                   f"{len(dup)} rows duplicate key {key}, e.g. {sample}")


def _check_return_units(df, report):
    if "total_return" not in df.columns:
        return
    for eid, grp in df.groupby("entity_id"):
        r = grp["total_return"].dropna()
        if r.empty:
            continue
        share = float((r.abs() > RETURN_ABS_WARN).mean())
        if share > RETURN_ABS_SHARE_LIMIT:
            report.add("returns_monthly", "unit_scale", "block",
                       f"{eid}: {share:.0%} of returns exceed "
                       f"{RETURN_ABS_WARN:.0%} in absolute value, "
                       f"max {r.abs().max():.2f}. Suspect percent values in a "
                       f"decimal column.")


def _check_referential(returns, securities, report):
    known = set(securities["entity_id"])
    orphans = sorted(set(returns["entity_id"]) - known)
    if orphans:
        report.add("returns_monthly", "referential", "block",
                   f"entity_ids absent from securities master: {orphans}")


def _check_calendar(returns, report, as_of=None, staleness_months=2):
    for eid, grp in returns.groupby("entity_id"):
        periods = pd.PeriodIndex(grp["period"].unique(), freq="M").sort_values()
        if len(periods) == 0:
            continue
        full = pd.period_range(periods.min(), periods.max(), freq="M")
        gaps = full.difference(periods)
        if len(gaps):
            shown = [str(p) for p in gaps[:4]]
            report.add("returns_monthly", "calendar_gap", "flag",
                       f"{eid}: {len(gaps)} interior month(s) missing, e.g. {shown}")
        if as_of is not None:
            lag = (pd.Period(as_of, freq="M") - periods.max()).n
            if lag > staleness_months:
                report.add("returns_monthly", "stale", "flag",
                           f"{eid}: last observation {periods.max()} is "
                           f"{lag} months before release as_of {as_of}")


def _check_cme(cme, corr, report):
    if "arithmetic_er" in cme.columns:
        bad = cme[cme["arithmetic_er"].abs() > CME_ER_ABS_WARN]
        if len(bad):
            report.add("cme_inputs", "unit_scale", "block",
                       f"{len(bad)} expected returns exceed "
                       f"{CME_ER_ABS_WARN:.0%}, likely percent not decimal")
    if "volatility" in cme.columns:
        lo, hi = CME_VOL_BOUNDS
        bad = cme[(cme["volatility"] < lo) | (cme["volatility"] > hi)]
        if len(bad):
            report.add("cme_inputs", "vol_bounds", "block",
                       f"{len(bad)} volatilities outside [{lo}, {hi}]")
    if "compound_er" in cme.columns and "arithmetic_er" in cme.columns:
        inverted = cme[cme["compound_er"] > cme["arithmetic_er"] + 1e-9]
        if len(inverted):
            report.add("cme_inputs", "er_ordering", "flag",
                       f"{len(inverted)} asset classes where compound ER exceeds "
                       f"arithmetic ER. Check the two columns are not swapped.")

    if corr is None:
        return
    M = corr.to_numpy(dtype=float)
    if M.shape[0] != M.shape[1]:
        report.add("cme_corr", "shape", "block", f"matrix is {M.shape}, not square")
        return
    if not np.allclose(M, M.T, atol=1e-8):
        report.add("cme_corr", "symmetry", "block", "correlation matrix is not symmetric")
    if not np.allclose(np.diag(M), 1.0, atol=1e-8):
        report.add("cme_corr", "diagonal", "block", "correlation diagonal is not 1")
    if (np.abs(M) > 1 + 1e-9).any():
        report.add("cme_corr", "bounds", "block", "correlation entries outside [-1, 1]")
    eig = np.linalg.eigvalsh((M + M.T) / 2)
    if eig.min() < -1e-8:
        report.add("cme_corr", "psd", "block",
                   f"matrix is not positive semidefinite, min eigenvalue "
                   f"{eig.min():.2e}. Optimizer results would be unreliable.")


def _check_factor_alignment(returns, factors, securities, report, min_obs=60):
    """Every fund flagged for a factor model needs min_obs overlapping months."""
    fac_periods = {
        fs: set(g["period"]) for fs, g in factors.groupby("factor_set")
    }
    eligible = securities[securities["equity"] & securities["factor_model"].notna()]
    for _, row in eligible.iterrows():
        eid = row["entity_id"]
        fs = row["factor_model"]
        r = returns[(returns["entity_id"] == eid) & returns["total_return"].notna()]
        if fs not in fac_periods:
            report.add("factors_monthly", "factor_set_missing", "block",
                       f"{eid} requires factor_set '{fs}' which is not in the release")
            continue
        overlap = len(set(r["period"]) & fac_periods[fs])
        if overlap < min_obs:
            report.add("returns_monthly", "min_observations", "flag",
                       f"{eid}: {overlap} usable months against {fs}, "
                       f"below the {min_obs}-month minimum for factor regression")


def _check_policy_weights(pw, securities, report, tol=1e-6):
    if "policy_weight" not in pw.columns:
        return
    total = float(pw["policy_weight"].sum())
    if abs(total - 1.0) > tol:
        report.add("policy_weights", "budget", "block",
                   f"weights sum to {total:.6f}, not 1. A portfolio that does not "
                   f"sum to one cannot be a starting point for the optimizer.")
    neg = pw[pw["policy_weight"] < 0]
    if len(neg):
        report.add("policy_weights", "long_only", "block",
                   f"{len(neg)} negative weights; Project 1 pilot is long-only")
    known = set(securities["entity_id"])
    orphans = sorted(set(pw["entity_id"]) - known)
    if orphans:
        report.add("policy_weights", "referential", "block",
                   f"weighted entity_ids absent from securities master: {orphans}")
    unweighted = sorted(known - set(pw["entity_id"]))
    if unweighted:
        report.add("policy_weights", "completeness", "flag",
                   f"funds in the master with no policy weight: {unweighted}")
    if "weight_type" in pw.columns and (pw["weight_type"] == "pending_client").any():
        report.add("policy_weights", "provenance", "flag",
                   "weight_type is pending_client. Strategic target and current "
                   "actual weights are different inputs; confirm which these are.")


def validate_release(tables, as_of=None, min_obs=60, staleness_months=2):
    report = ValidationReport()
    for name in ("securities", "returns_monthly", "factors_monthly", "cme_inputs"):
        if name not in tables:
            report.add(name, "presence", "block", "table missing from release")
            continue
        _check_schema(tables[name], name, report)
        _check_keys(tables[name], name, report)

    if "returns_monthly" in tables:
        _check_return_units(tables["returns_monthly"], report)
        _check_calendar(tables["returns_monthly"], report, as_of, staleness_months)
    if {"returns_monthly", "securities"} <= tables.keys():
        _check_referential(tables["returns_monthly"], tables["securities"], report)
    if "cme_inputs" in tables:
        _check_cme(tables["cme_inputs"], tables.get("cme_corr"), report)
    if {"policy_weights", "securities"} <= tables.keys():
        _check_policy_weights(tables["policy_weights"], tables["securities"], report)
    if {"returns_monthly", "factors_monthly", "securities"} <= tables.keys():
        _check_factor_alignment(tables["returns_monthly"], tables["factors_monthly"],
                                tables["securities"], report, min_obs)
    return report


# ---------------------------------------------------------------------------
# Coverage table. This is the artifact Teams 2 and 3 read before modelling.
# ---------------------------------------------------------------------------


def build_coverage(returns, securities, factors=None, min_obs=60):
    rows = []
    fac_periods = {}
    if factors is not None:
        fac_periods = {fs: set(g["period"]) for fs, g in factors.groupby("factor_set")}

    for _, sec in securities.iterrows():
        eid = sec["entity_id"]
        grp = returns[returns["entity_id"] == eid]
        obs = grp["total_return"].notna().sum()
        if len(grp):
            periods = pd.PeriodIndex(grp["period"].unique(), freq="M").sort_values()
            first, last = str(periods.min()), str(periods.max())
            span = (periods.max() - periods.min()).n + 1
            gaps = span - len(periods)
            missing_vals = int(grp["total_return"].isna().sum())
        else:
            first = last = None
            span = gaps = missing_vals = 0

        fs = sec["factor_model"]
        usable = len(set(grp.loc[grp["total_return"].notna(), "period"]) & fac_periods[fs]) \
            if fs in fac_periods else 0

        rows.append({
            "entity_id": eid,
            "asset_class": sec["asset_class"],
            "equity": bool(sec["equity"]),
            "first_obs": first,
            "last_obs": last,
            "months_spanned": span,
            "observations": int(obs),
            "interior_gaps": int(gaps),
            "missing_values": missing_vals,
            "factor_set": fs,
            "usable_factor_months": usable,
            "meets_min_obs": bool(usable >= min_obs) if sec["equity"] else None,
            "benchmark_proxy": sec["benchmark_proxy"],
            "ready_for": _readiness(sec, usable, min_obs),
        })
    return pd.DataFrame(rows)


def _readiness(sec, usable, min_obs):
    if not sec["equity"]:
        return "project_1_only"
    if usable >= min_obs:
        return "project_1_and_2"
    return "project_1_only_insufficient_history"


# ---------------------------------------------------------------------------
# Data dictionary and release manifest
# ---------------------------------------------------------------------------

FIELD_NOTES = {
    "entity_id": "Ticker used as the join key across every release table.",
    "period": "Calendar month as YYYY-MM string. Join key, never a timestamp.",
    "total_return": "Monthly total return as a DECIMAL, net of fund expenses.",
    "source_id": "Key into config/sources.yaml. Every observation is traceable.",
    "is_proxy": "True when the series is a labelled stand-in, not the fund itself.",
    "retrieved_at": "UTC date the observation was pulled, for reconstructing old reports.",
    "factor_set": "Which regional factor block the row belongs to.",
    "mkt_rf": "Market excess return, decimal. French file divided by 100.",
    "smb": "Small minus big, decimal.",
    "hml": "High minus low book-to-market, decimal.",
    "mom": "Momentum, decimal.",
    "rf": "Risk-free return, decimal. THE risk-free series for excess returns.",
    "arithmetic_er": "Annual arithmetic expected return, decimal. Optimizer input.",
    "compound_er": "Annual compound expected return, decimal. NOT an optimizer input.",
    "volatility": "Annual standard deviation, decimal.",
    "benchmark_proxy": "Proposed benchmark. Proxy until First Bank confirms.",
    "benchmark_is_proxy": "True whenever the benchmark was assigned by the team.",
    "liquid_designation": "Policy liquidity flag for the Project 1 liquid sleeve constraint.",
    "equity": "True when the fund is in scope for Project 2 factor attribution.",
    "factor_model": "Factor specification assigned to the fund.",
    "asset_class": "Project 1 optimization bucket.",
    "sleeve": "Finer grained role label, fund level.",
    "vintage": "CME publication vintage. Part of release identity.",
    "policy_weight": "First Bank portfolio weight, decimal. w0 in the turnover constraint.",
    "weight_type": "Strategic target or current actual. These are different model inputs.",
    "as_of": "Date the policy weights were struck.",
}


def build_dictionary(tables, conventions):
    lines = ["# Data dictionary", "",
             f"Release: `{conventions.get('release_id', 'unversioned')}`", ""]
    lines.append("## Conventions")
    for k in ("return_units", "period_key", "base_currency", "return_type",
              "fee_basis", "risk_free_source", "missing_policy"):
        if k in conventions:
            lines.append(f"- **{k}**: {conventions[k]}")
    lines.append("")
    for name, df in tables.items():
        if name not in SCHEMAS:
            continue
        spec = SCHEMAS[name]
        lines.append(f"## `{name}`")
        lines.append(f"Primary key: {', '.join(spec['key'])} | rows: {len(df):,}")
        lines.append("")
        lines.append("| column | type | notes |")
        lines.append("|---|---|---|")
        for col, kind in spec["columns"].items():
            note = FIELD_NOTES.get(col, "")
            lines.append(f"| `{col}` | {kind} | {note} |")
        lines.append("")
    return "\n".join(lines)


def build_manifest(paths, release_id, as_of, validation):
    entries = []
    for p in sorted(str(x) for x in paths):
        data = open(p, "rb").read()
        entries.append({
            "file": os.path.basename(p),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        })
    return {
        "release_id": release_id,
        "as_of": as_of,
        "validation_passed": validation.passed(),
        "blocking_issues": len(validation.blocking),
        "flags": len(validation.flags),
        "files": entries,
    }


def write_manifest(path, manifest):
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
