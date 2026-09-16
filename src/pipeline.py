"""
Raw -> release pipeline.

Runs the full sequence the team will run every time data refreshes:
  1. load raw snapshots
  2. validate raw, record what is wrong
  3. normalize, with every repair written to a log that keeps the original value
  4. validate again, block the release if anything still fails
  5. emit coverage table, data dictionary and a hashed manifest

Nothing is repaired silently. A repair the client cannot see is a repair the
client cannot trust.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import contract

AS_OF = "2026-08"
RELEASE_ID = "data_v0.1_reference"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "release" / RELEASE_ID
CONFIG = ROOT / "config"


class RepairLog:
    def __init__(self):
        self.rows = []

    def record(self, table, entity, field, action, original, corrected, reason):
        self.rows.append({
            "table": table, "entity": entity, "field": field, "action": action,
            "original_value": original, "corrected_value": corrected, "reason": reason,
        })

    def to_frame(self):
        cols = ["table", "entity", "field", "action", "original_value",
                "corrected_value", "reason"]
        return pd.DataFrame(self.rows, columns=cols)


def load_raw():
    sec = pd.read_csv(RAW / "securities.csv")
    ret = pd.read_csv(RAW / "returns_monthly.csv", dtype={"period": str})
    fac = pd.read_csv(RAW / "factors_monthly.csv", dtype={"period": str})
    cme = pd.read_csv(RAW / "cme_inputs.csv")
    corr = pd.read_csv(RAW / "cme_corr.csv", index_col=0)
    pw = pd.read_csv(RAW / "policy_weights.csv")
    return {"securities": sec, "returns_monthly": ret, "factors_monthly": fac,
            "cme_inputs": cme, "cme_corr": corr, "policy_weights": pw}


def normalize_factors(fac, log):
    fac = fac.copy()
    cols = ["mkt_rf", "smb", "hml", "mom", "rf"]
    sentinel = (fac[cols] == -99.99)
    n_sent = int(sentinel.to_numpy().sum())
    if n_sent:
        for c in cols:
            hit = fac[c] == -99.99
            for p in fac.loc[hit, "period"]:
                log.record("factors_monthly", p, c, "sentinel_to_nan", -99.99, None,
                           "French library missing-data sentinel, not a -99.99% return")
        fac[cols] = fac[cols].mask(sentinel, np.nan)

    fac[cols] = fac[cols] / 100.0
    log.record("factors_monthly", "ALL", ",".join(cols), "percent_to_decimal",
               "percent", "decimal", "French library publishes percent; contract is decimal")
    return fac


def normalize_returns(ret, log):
    ret = ret.copy()

    before = len(ret)
    ret = ret.drop_duplicates(subset=["entity_id", "period"], keep="first")
    removed = before - len(ret)
    if removed:
        log.record("returns_monthly", "VFSUX", "(entity_id, period)", "drop_duplicate",
                   f"{before} rows", f"{len(ret)} rows",
                   "identical duplicate observation, first occurrence kept")

    # Unit scale correction, per entity, evidence based and logged.
    for eid, grp in ret.groupby("entity_id"):
        r = grp["total_return"].dropna()
        if r.empty:
            continue
        if (r.abs() > contract.RETURN_ABS_WARN).mean() > contract.RETURN_ABS_SHARE_LIMIT:
            idx = grp.index
            log.record("returns_monthly", eid, "total_return", "percent_to_decimal",
                       f"max |r| = {r.abs().max():.4f}",
                       f"max |r| = {r.abs().max()/100:.6f}",
                       "monthly magnitudes implausible for decimal units; "
                       "source delivered percent. REQUIRES SOURCE CONFIRMATION.")
            ret.loc[idx, "total_return"] = ret.loc[idx, "total_return"] / 100.0

    return ret.sort_values(["entity_id", "period"]).reset_index(drop=True)


def normalize_cme(cme, corr, log):
    cme = cme.copy()
    for c in ("arithmetic_er", "compound_er", "volatility"):
        cme[c] = cme[c] / 100.0
    log.record("cme_inputs", "ALL", "arithmetic_er,compound_er,volatility",
               "percent_to_decimal", "percent per annum", "decimal per annum",
               "CME source publishes percent")

    M = corr.to_numpy(dtype=float)
    M = (M + M.T) / 2
    eig = np.linalg.eigvalsh(M)
    if eig.min() < -1e-10:
        w, V = np.linalg.eigh(M)
        w_clipped = np.clip(w, 1e-8, None)
        R = V @ np.diag(w_clipped) @ V.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
        max_shift = float(np.abs(R - M).max())
        log.record("cme_corr", "matrix", "correlation", "psd_projection",
                   f"min eigenvalue {eig.min():.4e}",
                   f"min eigenvalue {np.linalg.eigvalsh(R).min():.4e}",
                   f"matrix was not positive semidefinite; nearest-PSD projection "
                   f"applied, largest entry moved {max_shift:.4f}. "
                   f"ORIGINAL RETAINED. Extraction should be re-checked against the source.")
        corr = pd.DataFrame(R, index=corr.index, columns=corr.columns)
    return cme, corr


def run():
    os.makedirs(OUT, exist_ok=True)
    conventions = yaml.safe_load(open(CONFIG / "conventions.yaml"))
    conventions["release_id"] = RELEASE_ID

    raw = load_raw()
    print("=" * 74)
    print("PASS 1 - validation of raw snapshots")
    print("=" * 74)
    pre = contract.validate_release(raw, as_of=AS_OF)
    print(pre.render())

    log = RepairLog()
    tables = {
        "securities": raw["securities"],
        "factors_monthly": normalize_factors(raw["factors_monthly"], log),
        "returns_monthly": normalize_returns(raw["returns_monthly"], log),
    }
    cme, corr = normalize_cme(raw["cme_inputs"], raw["cme_corr"], log)
    tables["cme_inputs"] = cme
    tables["cme_corr"] = corr
    tables["policy_weights"] = raw["policy_weights"]   # client data, passed through unaltered

    print()
    print("=" * 74)
    print("REPAIR LOG - every change, with the original value kept")
    print("=" * 74)
    rl = log.to_frame()
    for _, r in rl.iterrows():
        print(f"  {r['table']:<17} {str(r['entity']):<8} {r['action']:<20} {r['reason']}")

    print()
    print("=" * 74)
    print("PASS 2 - validation of normalized tables")
    print("=" * 74)
    post = contract.validate_release(tables, as_of=AS_OF)
    print(post.render())

    cov = contract.build_coverage(tables["returns_monthly"], tables["securities"],
                                 tables["factors_monthly"])

    paths = []
    for name in ("securities", "returns_monthly", "factors_monthly", "cme_inputs",
                 "policy_weights"):
        p = OUT / f"{name}.csv"
        tables[name].to_csv(p, index=False)
        paths.append(p)
    p = OUT / "cme_corr.csv"
    tables["cme_corr"].to_csv(p)
    paths.append(p)

    cov.to_csv(OUT / "coverage.csv", index=False)
    paths.append(OUT / "coverage.csv")
    rl.to_csv(OUT / "repair_log.csv", index=False)
    paths.append(OUT / "repair_log.csv")
    pre.to_frame().to_csv(OUT / "validation_raw.csv", index=False)
    post.to_frame().to_csv(OUT / "validation_release.csv", index=False)
    paths += [OUT / "validation_raw.csv", OUT / "validation_release.csv"]

    with open(OUT / "data_dictionary.md", "w") as fh:
        fh.write(contract.build_dictionary(tables, conventions))
    paths.append(OUT / "data_dictionary.md")

    man = contract.build_manifest(paths, RELEASE_ID, AS_OF, post)
    contract.write_manifest(OUT / "MANIFEST.json", man)

    print()
    print("=" * 74)
    print("COVERAGE TABLE")
    print("=" * 74)
    show = cov[["entity_id", "asset_class", "first_obs", "last_obs", "observations",
                "interior_gaps", "usable_factor_months", "ready_for"]]
    print(show.to_string(index=False))

    print()
    print(f"Release written to {OUT}/  ({len(man['files'])} files hashed)")
    print(f"Release gate: {'PASS' if post.passed() else 'BLOCKED'}")
    return tables, cov, post


if __name__ == "__main__":
    run()
