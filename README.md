# First Bank practicum — Team 1, Data Foundation

Owner: Pranjay (Team 1 lead)
Release in this repo: `data_v0.1_reference`
Status: pre-client-meeting working package, September 16 2026

## What this is

The official data layer for Projects 1 (Asset Allocation) and 2 (Factor Attribution).
Teams 2, 3 and 4 code against the tables and conventions defined here. Swapping
reference data for client data changes the release id, not their code.

## READ THIS FIRST

The data in `release/data_v0.1_reference/` is **synthetic**. It is simulated from a
known factor structure so the pipeline and the downstream models can be built and
tested before any real fund data exists. **No number in it is a real First Bank
fund return and none of it may be shown to the client as a result.**

Its purpose is the opposite: it is a test harness. Six defects we expect to hit
with the real files are injected deliberately, and the validation gates catch
all six.

## Contract tables

| table | key | holds |
|---|---|---|
| `securities` | `entity_id` | 13 funds, asset-class map, benchmark proxies, Project 2 eligibility |
| `returns_monthly` | `entity_id`, `period` | monthly total returns, decimal, with source and retrieval date |
| `factors_monthly` | `period`, `factor_set` | US, developed and emerging factor blocks plus RF |
| `cme_inputs` | `asset_class` | arithmetic ER, compound ER, volatility per asset class |
| `cme_corr` | matrix | asset-class correlation matrix |
| `policy_weights` | `entity_id` | First Bank portfolio weights, w0 for the turnover constraint |
| `coverage` | `entity_id` | what each fund is actually usable for |
| `repair_log` | — | every correction with the original value retained |
| `MANIFEST.json` | — | SHA256 per file, validation status |

Conventions live in `config/conventions.yaml` and are binding. The short version:
returns are **decimals**, `period` is a **`YYYY-MM` string**, the risk-free rate
is the **French RF column** and never FRED, and missing months stay missing.

## Run it

Windows (PowerShell or cmd):

```
python src\make_reference.py     # build the reference data, creates raw/
python src\pipeline.py           # raw -> validated release
python src\demo.py               # prove both projects consume it
python run_tests.py              # 15 known-answer tests, no pytest needed
```

macOS or Linux:

```bash
python3 src/make_reference.py
python3 src/pipeline.py
python3 src/demo.py
python3 run_tests.py
```

Run them in that order the first time; `demo.py` reads what `pipeline.py` writes.
All paths resolve from the project root, so the working directory does not
matter. `raw/` and `release/` are created automatically.

With network and a source decision:

```bash
python3 src/ingest.py french                  # Kenneth French factor blocks
python3 src/ingest.py fred                    # rate context for Project 5
python3 src/ingest.py returns --input export.csv --layout wrds
```

## What the universe tells us

The 13-fund 60/40 Select list splits **9 equity, 4 fixed income**, and maps onto
exactly **8 asset classes**, which sits inside Project 1's stated 6–8 pilot scope.

That leaves one scope gap: Project 2's pilot calls for 10–15 equity funds and the
Approved List subset supplies 9, one of which may fail the 60-month floor
depending on share-class inception.

## The policy portfolio

First Bank supplied the 60/40 Select weights on September 16. They sum to exactly
1.0000 and equity sums to exactly 60.00%, so these are struck targets rather than
drifted holdings, but the label still needs confirming.

Aggregated to asset classes: US large 36%, core bond 20%, international developed
16%, short IG credit 8%, inflation linked 6%, international bond 6%, US small 5%,
emerging markets 3%.

Two things fall out immediately:

- **US large equity holds 36% of capital and drives 56% of portfolio risk** under
  the reference CME matrix. Concentration of this kind is invisible in a weights
  table, which is why the allocation tool reports risk contributions.
- **A plausible 30% per-class cap makes the live portfolio non-compliant.** US
  large at 36% breaches it, and reaching compliance costs a minimum 6% one-way
  turnover. Our placeholder constraint, not First Bank's policy, but it means the
  real limits are a blocking input rather than a nice-to-have.

## Open questions for First Bank

1. Is the 13-fund list the pilot universe, or a subset of a larger Approved List?
   Project 2 needs 10–15 equity funds; we have 9.
2. Are share classes fixed? Institutional and Admiral classes are often younger
   than the strategy, and the newest one bounds any common sample window.
3. Preferred return source. Does First Bank have one, or do we use WRDS/CRSP via
   WashU? This is the largest single schedule risk in the data workstream.
4. Official benchmark per fund, or do our proxies stand? Our assignment drives
   every style-drift alert in Project 2.
5. Which CME vintage, and does First Bank have its own capital market
   expectations to use instead of a public provider?
6. Real policy constraints for Project 1: per-class min and max, the liquid sleeve
   definition, turnover limit. Blocking input. Our placeholder 30% class cap makes
   the supplied portfolio infeasible, so we cannot guess these.
7. Are the supplied weights strategic targets or current actual holdings? Targets
   are w0 for the optimizer; actual holdings imply a drift and rebalancing question
   instead. Currently `pending_client` in `universe.yaml`.
8. Rebalancing policy and tolerance bands, and whether turnover is constrained.
9. History window preference, and whether daily data is available for later
   drawdown and stress work.
10. Known style-drift or manager-change episodes we can use as labelled test cases.

## Ownership and use

Coursework produced for First Bank under the WashU Quantitative Finance
Practicum. Not licensed for redistribution or reuse. Intellectual property and
handover terms follow the course and client agreement, not any open-source
license, which is why this repository carries no LICENSE file.

Repository is private. Data releases are not stored here (see `.gitignore`);
they live in the shared team drive and are identified by `release_id` with
`MANIFEST.json` as the integrity record.

## Environment note

Built and tested on Python 3.12 with pandas, numpy and scipy only. The target
stack adds statsmodels (Team 3 regressions), CVXPY (Team 2 optimizer), pyarrow
for Parquet snapshots and DuckDB for joins. The contract does not depend on any
of them.
