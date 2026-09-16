"""
Real source adapters. Each writes a dated raw snapshot, then the same
pipeline.py normalization and validation applies.

These need network access and, for fund returns, a source decision from the
practicum lead and First Bank. Everything here is written against the documented
quirks of each source rather than the happy path, because the quirks are what
cost time.

Usage:
    python3 src/ingest.py french   --outdir raw
    python3 src/ingest.py fred     --outdir raw
    python3 src/ingest.py returns  --input path/to/export.csv --layout wrds
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

FRENCH_FILES = {
    # factor_set -> (zip filename, csv columns after the date column)
    "ff3_mom": ("F-F_Research_Data_Factors_CSV.zip", ["mkt_rf", "smb", "hml", "rf"]),
    "ff3_mom_momentum": ("F-F_Momentum_Factor_CSV.zip", ["mom"]),
    "ff3_mom_developed": ("Developed_3_Factors_CSV.zip", ["mkt_rf", "smb", "hml", "rf"]),
    "ff3_mom_developed_mom": ("Developed_Mom_Factor_CSV.zip", ["mom"]),
    "ff3_mom_emerging": ("Emerging_5_Factors_CSV.zip",
                         ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]),
}

SENTINELS = [-99.99, -999]


def _parse_french_csv(text, value_cols):
    """
    French CSVs carry a header preamble, then a monthly block keyed YYYYMM, then
    an annual block keyed YYYY, sometimes with further copyright lines. Slicing
    on the first blank line is the usual mistake. Key on key length instead.
    """
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        if len(parts[0]) != 6:          # 6 digits = monthly, 4 = annual block
            continue
        try:
            vals = [float(p) for p in parts[1:1 + len(value_cols)]]
        except ValueError:
            continue
        rows.append([parts[0]] + vals)

    df = pd.DataFrame(rows, columns=["yyyymm"] + value_cols)
    df["period"] = df["yyyymm"].str[:4] + "-" + df["yyyymm"].str[4:6]
    df = df.drop(columns="yyyymm")
    for s in SENTINELS:
        df[value_cols] = df[value_cols].replace(s, np.nan)
    # NOTE: values stay in PERCENT here. pipeline.normalize_factors converts.
    return df[["period"] + value_cols]


def fetch_french(outdir="raw"):
    import requests

    blocks = {}
    for key, (fname, cols) in FRENCH_FILES.items():
        url = f"{FRENCH_BASE}/{fname}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("latin-1")
        blocks[key] = _parse_french_csv(text, cols)
        print(f"  {key}: {len(blocks[key])} monthly rows from {fname}")

    frames = []
    for fs, mom_key in (("ff3_mom", "ff3_mom_momentum"),
                        ("ff3_mom_developed", "ff3_mom_developed_mom")):
        base = blocks[fs]
        mom = blocks[mom_key]
        merged = base.merge(mom, on="period", how="left")
        merged["factor_set"] = fs
        merged["source_id"] = ("french_ff5_mom" if fs == "ff3_mom"
                               else "french_developed")
        frames.append(merged)

    em = blocks["ff3_mom_emerging"].copy()
    em["mom"] = np.nan          # emerging momentum is a separate file if needed
    em["factor_set"] = "ff3_mom_emerging"
    em["source_id"] = "french_developed"
    frames.append(em)

    out = pd.concat(frames, ignore_index=True)
    keep = ["period", "factor_set", "mkt_rf", "smb", "hml", "mom", "rf", "source_id"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep].sort_values(["factor_set", "period"])
    path = f"{outdir}/factors_monthly.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out):,} rows, PERCENT units, pipeline converts)")
    return out


def fetch_fred(series=("DGS10", "DGS2", "CPIAUCSL"), outdir="raw"):
    """Monthly average of each series. Rate CONTEXT for Project 5, not returns."""
    import requests

    frames = []
    for s in series:
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={s}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        date_col, val_col = df.columns[0], df.columns[1]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df["period"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)
        g = df.groupby("period", as_index=False)[val_col].mean()
        g.columns = ["period", "value"]
        g["series_id"] = s
        g["source_id"] = "fred"
        frames.append(g)
        print(f"  {s}: {len(g)} monthly observations")

    out = pd.concat(frames, ignore_index=True)[
        ["period", "series_id", "value", "source_id"]]
    path = f"{outdir}/macro_monthly.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    return out


# ---------------------------------------------------------------------------
# Fund returns. Layout depends on the source decision still open with Ron.
# ---------------------------------------------------------------------------

LAYOUTS = {
    "wrds": dict(entity="ticker", period="caldt", value="mret",
                 units="decimal",
                 note="CRSP Mutual Fund monthly returns. Preferred if WashU "
                      "entitlement covers it. mret is already a total return."),
    "morningstar": dict(entity="Ticker", period="Date", value="Monthly Return",
                        units="percent",
                        note="Morningstar Direct export. Returns come as percent."),
    "issuer_nav": dict(entity="ticker", period="date", value=None,
                       units="derived",
                       note="Reconstruct from NAV plus distributions. Highest "
                            "effort path and the largest schedule risk."),
}


def load_fund_returns(path, layout="wrds", outdir="raw"):
    spec = LAYOUTS[layout]
    print(f"layout '{layout}': {spec['note']}")
    df = pd.read_csv(path)

    if layout == "issuer_nav":
        return _reconstruct_from_nav(df, outdir)

    out = pd.DataFrame({
        "entity_id": df[spec["entity"]].astype(str).str.upper().str.strip(),
        "period": pd.to_datetime(df[spec["period"]]).dt.to_period("M").astype(str),
        "total_return": pd.to_numeric(df[spec["value"]], errors="coerce"),
        "source_id": f"fund_returns_{layout}",
        "is_proxy": False,
        "retrieved_at": dt.date.today().isoformat(),
    })
    # Units are NOT converted here. pipeline.normalize_returns detects and logs
    # the conversion so the decision is visible in the repair log.
    p = f"{outdir}/returns_monthly.csv"
    out.to_csv(p, index=False)
    print(f"wrote {p}  ({len(out):,} rows, source units '{spec['units']}')")
    return out


def _reconstruct_from_nav(df, outdir):
    """
    total_return_t = (NAV_t + distributions_t) / NAV_{t-1} - 1

    Only valid when the distribution record is COMPLETE. A missed capital gains
    distribution understates the return in that month, which shows up as
    spurious negative alpha. Every fund reconstructed this way must be
    reconciled against a published annual return before release.
    """
    need = {"ticker", "date", "nav", "distribution"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"issuer_nav layout requires columns {sorted(need)}, "
                         f"missing {sorted(missing)}")
    df = df.copy()
    df["period"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    df = df.sort_values(["ticker", "period"])
    df["distribution"] = df["distribution"].fillna(0.0)
    df["prev_nav"] = df.groupby("ticker")["nav"].shift(1)
    df["total_return"] = (df["nav"] + df["distribution"]) / df["prev_nav"] - 1

    out = df.dropna(subset=["total_return"])[["ticker", "period", "total_return"]]
    out = out.rename(columns={"ticker": "entity_id"})
    out["source_id"] = "fund_returns_issuer_nav"
    out["is_proxy"] = False
    out["retrieved_at"] = dt.date.today().isoformat()
    p = f"{outdir}/returns_monthly.csv"
    out.to_csv(p, index=False)
    print(f"wrote {p}  ({len(out):,} rows reconstructed)")
    print("REQUIRED NEXT STEP: reconcile each fund against a published annual "
          "return before this release is tagged.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["french", "fred", "returns"])
    ap.add_argument("--outdir", default="raw")
    ap.add_argument("--input")
    ap.add_argument("--layout", default="wrds", choices=list(LAYOUTS))
    a = ap.parse_args()

    if a.command == "french":
        fetch_french(a.outdir)
    elif a.command == "fred":
        fetch_fred(outdir=a.outdir)
    else:
        if not a.input:
            ap.error("returns requires --input")
        load_fund_returns(a.input, a.layout, a.outdir)


if __name__ == "__main__":
    main()
