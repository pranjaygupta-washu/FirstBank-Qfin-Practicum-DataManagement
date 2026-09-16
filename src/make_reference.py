"""
Generates the SYNTHETIC reference data package.

These are NOT real First Bank fund returns. The numbers are simulated from a
known factor structure so that (a) Teams 2 and 3 can build and test against the
real column contract before any client data exists, and (b) every validation
gate has something to catch.

Six defects are injected on purpose, each one a failure mode we expect to hit
with the real files:

  1. DODIX returns stored in PERCENT while the column says decimal
  2. OBSIX share-class history only 56 months, below the 60-month regression floor
  3. DFEMX missing two interior months
  4. VFSUX one duplicated (entity_id, period) row
  5. DFAIX last observation stale by four months
  6. CME correlation matrix not positive semidefinite, plus a French -99.99 sentinel

Regenerate deterministically with SEED. Nothing here is a forecast.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Paths resolve from the project root, not the current working directory, so the
# scripts run the same way from anywhere and on any operating system.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
CONFIG_DIR = ROOT / "config"

SEED = 20260916
START = "2011-01"
END = "2026-08"
RETRIEVED = "2026-09-16"

FACTOR_SETS = ["ff3_mom", "ff3_mom_developed", "ff3_mom_emerging"]

# Betas used to simulate each fund. Chosen to look like the stated style so the
# demo report cards are interpretable, not to represent the actual funds.
BETAS = {
    "FAGCX": dict(mkt=1.08, smb=0.15, hml=-0.30, mom=0.12, alpha=0.0010),
    "VFIAX": dict(mkt=1.00, smb=-0.03, hml=0.00, mom=0.00, alpha=-0.0002),
    "DODGX": dict(mkt=1.02, smb=0.10, hml=0.38, mom=-0.08, alpha=0.0004),
    "OBSIX": dict(mkt=1.15, smb=0.85, hml=-0.35, mom=0.25, alpha=0.0008),
    "DFFVX": dict(mkt=1.05, smb=0.80, hml=0.50, mom=-0.10, alpha=0.0001),
    "FIGFX": dict(mkt=0.95, smb=0.05, hml=-0.25, mom=0.10, alpha=0.0006),
    "VTMGX": dict(mkt=0.99, smb=0.08, hml=0.05, mom=0.00, alpha=-0.0001),
    "DODFX": dict(mkt=1.04, smb=0.12, hml=0.42, mom=-0.12, alpha=0.0002),
    "DFEMX": dict(mkt=0.98, smb=0.18, hml=0.20, mom=-0.05, alpha=-0.0003),
}
FI_PROFILE = {
    "VFSUX": dict(vol=0.0060, drift=0.0022),
    "DODIX": dict(vol=0.0110, drift=0.0028),
    "DFAIX": dict(vol=0.0075, drift=0.0021),
    "BNDX": dict(vol=0.0090, drift=0.0024),
}

# Style drift case: FAGCX value exposure walks from +0.10 to -0.45 across the
# sample, so Team 3's rolling regression and drift alert have a known signal.
DRIFT_FUND = "FAGCX"
DRIFT_HML = (0.10, -0.45)

CME_RAW_PERCENT = {
    # asset_class: (arithmetic_er, compound_er, volatility) in PERCENT per annum
    "us_large_equity": (7.9, 6.7, 16.2),
    "us_small_equity": (8.6, 6.9, 20.5),
    "intl_developed_equity": (8.4, 7.2, 17.4),
    "em_equity": (9.3, 7.4, 22.8),
    "short_ig_credit": (4.3, 4.2, 3.1),
    "core_bond": (4.9, 4.8, 5.4),
    "inflation_linked": (4.1, 4.0, 4.2),
    "intl_bond_hedged": (4.2, 4.1, 4.6),
}
CME_VINTAGE = "synthetic_reference_vintage"


def periods(start=START, end=END):
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def make_factors(rng):
    rows = []
    for fs in FACTOR_SETS:
        scale = {"ff3_mom": 1.0, "ff3_mom_developed": 1.05, "ff3_mom_emerging": 1.30}[fs]
        p = periods()
        n = len(p)
        rows.append(pd.DataFrame({
            "period": p,
            "factor_set": fs,
            # stored in PERCENT, as the French files arrive
            "mkt_rf": rng.normal(0.62, 4.35 * scale, n).round(2),
            "smb": rng.normal(0.10, 2.60, n).round(2),
            "hml": rng.normal(0.05, 2.90, n).round(2),
            "mom": rng.normal(0.18, 3.10, n).round(2),
            "rf": np.clip(rng.normal(0.14, 0.13, n), 0, None).round(2),
            "source_id": "french_ff5_mom" if fs == "ff3_mom" else "french_developed",
        }))
    fac = pd.concat(rows, ignore_index=True)

    # DEFECT 6a: French missing sentinel left in one developed-market row
    mask = (fac["factor_set"] == "ff3_mom_developed") & (fac["period"] == "2011-03")
    fac.loc[mask, ["smb", "hml", "mom"]] = -99.99
    return fac


def make_returns(fac, rng):
    fac_dec = fac.copy()
    for c in ("mkt_rf", "smb", "hml", "mom", "rf"):
        fac_dec[c] = fac_dec[c].replace(-99.99, np.nan) / 100.0

    fs_for = {
        "FAGCX": "ff3_mom", "VFIAX": "ff3_mom", "DODGX": "ff3_mom",
        "OBSIX": "ff3_mom", "DFFVX": "ff3_mom",
        "FIGFX": "ff3_mom_developed", "VTMGX": "ff3_mom_developed",
        "DODFX": "ff3_mom_developed", "DFEMX": "ff3_mom_emerging",
    }

    frames = []
    for eid, b in BETAS.items():
        f = fac_dec[fac_dec["factor_set"] == fs_for[eid]].reset_index(drop=True)
        n = len(f)
        hml_beta = np.full(n, b["hml"])
        if eid == DRIFT_FUND:
            hml_beta = np.linspace(DRIFT_HML[0], DRIFT_HML[1], n)
        resid_vol = 0.012 if eid in ("VFIAX", "VTMGX") else 0.022
        r = (f["rf"].fillna(0)
             + b["alpha"]
             + b["mkt"] * f["mkt_rf"].fillna(0)
             + b["smb"] * f["smb"].fillna(0)
             + hml_beta * f["hml"].fillna(0)
             + b["mom"] * f["mom"].fillna(0)
             + rng.normal(0, resid_vol, n))
        frames.append(pd.DataFrame({
            "entity_id": eid,
            "period": f["period"],
            "total_return": r.round(6),
            "source_id": "fund_returns_primary",
            "is_proxy": False,
            "retrieved_at": RETRIEVED,
        }))

    for eid, prof in FI_PROFILE.items():
        p = periods()
        n = len(p)
        r = rng.normal(prof["drift"], prof["vol"], n)
        frames.append(pd.DataFrame({
            "entity_id": eid,
            "period": p,
            "total_return": r.round(6),
            "source_id": "fund_returns_primary",
            "is_proxy": False,
            "retrieved_at": RETRIEVED,
        }))

    ret = pd.concat(frames, ignore_index=True)

    # DEFECT 1: DODIX arrives in percent
    m = ret["entity_id"] == "DODIX"
    ret.loc[m, "total_return"] = (ret.loc[m, "total_return"] * 100).round(4)

    # DEFECT 2: OBSIX institutional share class launched 2022-01 -> 56 months
    ret = ret[~((ret["entity_id"] == "OBSIX") & (ret["period"] < "2022-01"))]

    # DEFECT 3: DFEMX two interior months absent
    ret = ret[~((ret["entity_id"] == "DFEMX") & (ret["period"].isin(["2019-03", "2019-04"])))]

    # DEFECT 4: duplicated VFSUX row
    dup = ret[(ret["entity_id"] == "VFSUX") & (ret["period"] == "2018-07")]
    ret = pd.concat([ret, dup], ignore_index=True)

    # DEFECT 5: DFAIX stale, stops 2026-02
    ret = ret[~((ret["entity_id"] == "DFAIX") & (ret["period"] > "2026-02"))]

    return ret.sort_values(["entity_id", "period"]).reset_index(drop=True)


def make_securities(universe):
    rows = []
    for f in universe["funds"]:
        rows.append({
            "entity_id": f["ticker"],
            "legal_name": f["legal_name"],
            "asset_class": f["asset_class"],
            "sleeve": f["sleeve"],
            "region": f["region"],
            "style": f["style"],
            "equity": bool(f["equity"]),
            "factor_model": f["factor_model"],
            "benchmark_proxy": f["benchmark_proxy"],
            "benchmark_is_proxy": True,
            "liquid_designation": f["liquid_designation"],
        })
    return pd.DataFrame(rows)


def make_policy_weights(universe):
    """Real client weights, carried through as data. Not simulated."""
    pp = universe.get("policy_portfolio", {})
    return pd.DataFrame([{
        "entity_id": f["ticker"],
        "policy_weight": f["policy_weight"],
        "asset_class": f["asset_class"],
        "weight_type": pp.get("weight_type", "pending_client"),
        "as_of": pp.get("as_of", "pending_client"),
        "source_id": "first_bank_supplied",
    } for f in universe["funds"]])


def make_cme(rng, classes):
    rows = [{
        "asset_class": c,
        "vintage": CME_VINTAGE,
        "arithmetic_er": CME_RAW_PERCENT[c][0],
        "compound_er": CME_RAW_PERCENT[c][1],
        "volatility": CME_RAW_PERCENT[c][2],
        "source_id": "jpm_ltcma",
    } for c in classes]
    cme = pd.DataFrame(rows)  # PERCENT units, as extracted from the PDF

    base = np.array([
        [1.00, 0.88, 0.80, 0.72, 0.10, 0.02, 0.08, 0.05],
        [0.88, 1.00, 0.72, 0.68, 0.08, -0.02, 0.06, 0.02],
        [0.80, 0.72, 1.00, 0.78, 0.14, 0.06, 0.12, 0.10],
        [0.72, 0.68, 0.78, 1.00, 0.18, 0.04, 0.14, 0.12],
        [0.10, 0.08, 0.14, 0.18, 1.00, 0.62, 0.48, 0.44],
        [0.02, -0.02, 0.06, 0.04, 0.62, 1.00, 0.55, 0.70],
        [0.08, 0.06, 0.12, 0.14, 0.48, 0.55, 1.00, 0.50],
        [0.05, 0.02, 0.10, 0.12, 0.44, 0.70, 0.50, 1.00],
    ])
    # DEFECT 6b: transcription slip breaks positive semidefiniteness
    base[0, 1] = base[1, 0] = 0.995
    base[0, 2] = base[2, 0] = 0.97
    corr = pd.DataFrame(base, index=classes, columns=classes)
    return cme, corr


def main(outdir=None, cfgdir=None):
    outdir = Path(outdir) if outdir else RAW_DIR
    cfgdir = Path(cfgdir) if cfgdir else CONFIG_DIR
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    universe = yaml.safe_load(open(cfgdir / "universe.yaml"))
    classes = [c["id"] for c in universe["asset_classes"]]

    fac = make_factors(rng)
    ret = make_returns(fac, rng)
    sec = make_securities(universe)
    cme, corr = make_cme(rng, classes)
    pw = make_policy_weights(universe)

    sec.to_csv(outdir / "securities.csv", index=False)
    ret.to_csv(outdir / "returns_monthly.csv", index=False)
    fac.to_csv(outdir / "factors_monthly.csv", index=False)
    cme.to_csv(outdir / "cme_inputs.csv", index=False)
    corr.to_csv(outdir / "cme_corr.csv")
    pw.to_csv(outdir / "policy_weights.csv", index=False)

    print(f"raw written to {outdir}")
    print(f"  {len(sec)} securities, {len(ret):,} return rows, "
          f"{len(fac):,} factor rows, {len(cme)} asset classes")
    print("SYNTHETIC DATA. Not real fund returns. Six defects injected on purpose.")


if __name__ == "__main__":
    main()
