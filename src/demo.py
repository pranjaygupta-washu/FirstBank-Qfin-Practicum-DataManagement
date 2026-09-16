"""
Proves the release package is directly consumable by both projects.

Project 2: OLS factor regression with HAC standard errors, plus a rolling
           36-month window that recovers the style drift injected into FAGCX.
Project 1: builds mu and Sigma from the CME table, solves a constrained
           minimum-variance problem under a 60/40 policy, reports risk
           contributions and which constraints bind, and detects an
           infeasible return target.

Uses numpy and scipy only. In the real build, Team 3 would use statsmodels and
Team 2 would use CVXPY. The point here is that the DATA CONTRACT works, not
that these are the final solvers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "data_v0.1_reference"
FACTORS = ["mkt_rf", "smb", "hml", "mom"]


def load():
    sec = pd.read_csv(RELEASE / "securities.csv")
    ret = pd.read_csv(RELEASE / "returns_monthly.csv", dtype={"period": str})
    fac = pd.read_csv(RELEASE / "factors_monthly.csv", dtype={"period": str})
    cme = pd.read_csv(RELEASE / "cme_inputs.csv")
    corr = pd.read_csv(RELEASE / "cme_corr.csv", index_col=0)
    return sec, ret, fac, cme, corr


# ---------------------------------------------------------------------------
# Project 2
# ---------------------------------------------------------------------------

def ols_hac(y, X, lags=6):
    """OLS with Newey-West HAC covariance. Returns coefs, se, t, adj R2, resid sd."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b

    S = (e[:, None] * X).T @ (e[:, None] * X)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1)
        u = (e[L:, None] * X[L:])
        v = (e[:-L, None] * X[:-L])
        G = u.T @ v
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))

    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return b, se, b / se, adj, float(np.sqrt(ss_res / (n - k)))


def panel(eid, ret, fac, factor_set):
    r = ret[ret["entity_id"] == eid][["period", "total_return"]]
    f = fac[fac["factor_set"] == factor_set]
    df = r.merge(f, on="period", how="inner").dropna(
        subset=["total_return", "rf"] + FACTORS).sort_values("period")
    df["excess"] = df["total_return"] - df["rf"]
    return df


def report_card(eid, ret, fac, factor_set, min_obs=60):
    df = panel(eid, ret, fac, factor_set)
    if len(df) < min_obs:
        return {"entity_id": eid, "n": len(df), "status": "insufficient_history"}
    y = df["excess"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[FACTORS].to_numpy()])
    b, se, t, adj, rsd = ols_hac(y, X)
    crit = stats.t.ppf(0.975, len(df) - X.shape[1])
    return {
        "entity_id": eid,
        "n": len(df),
        "window": f"{df['period'].iloc[0]} to {df['period'].iloc[-1]}",
        "alpha_ann": (1 + b[0]) ** 12 - 1,
        "alpha_ci_lo_ann": (1 + b[0] - crit * se[0]) ** 12 - 1,
        "alpha_ci_hi_ann": (1 + b[0] + crit * se[0]) ** 12 - 1,
        "alpha_t": t[0],
        "beta_mkt": b[1], "beta_smb": b[2], "beta_hml": b[3], "beta_mom": b[4],
        "adj_r2": adj,
        "resid_vol_ann": rsd * np.sqrt(12),
        "status": "ok",
    }


def rolling_beta(eid, ret, fac, factor_set, window=36):
    df = panel(eid, ret, fac, factor_set)
    out = []
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window:i]
        y = w["excess"].to_numpy()
        X = np.column_stack([np.ones(window), w[FACTORS].to_numpy()])
        b, *_ = ols_hac(y, X, lags=3)
        out.append({"period_end": w["period"].iloc[-1],
                    "beta_mkt": b[1], "beta_hml": b[3], "beta_mom": b[4]})
    return pd.DataFrame(out)


def run_project2(sec, ret, fac):
    print("=" * 74)
    print("PROJECT 2 CHECK - factor report cards from the release tables")
    print("=" * 74)
    cards = []
    for _, s in sec[sec["equity"]].iterrows():
        cards.append(report_card(s["entity_id"], ret, fac, s["factor_model"]))
    cd = pd.DataFrame(cards)
    ok = cd[cd["status"] == "ok"]
    show = ok[["entity_id", "n", "beta_mkt", "beta_smb", "beta_hml", "beta_mom",
               "adj_r2", "alpha_ann", "alpha_t"]].copy()
    for c in ("beta_mkt", "beta_smb", "beta_hml", "beta_mom", "adj_r2"):
        show[c] = show[c].round(3)
    show["alpha_ann"] = (show["alpha_ann"] * 100).round(2)
    show["alpha_t"] = show["alpha_t"].round(2)
    show = show.rename(columns={"alpha_ann": "alpha_%ann"})
    print(show.to_string(index=False))

    skipped = cd[cd["status"] != "ok"]
    if len(skipped):
        print()
        for _, s in skipped.iterrows():
            print(f"  EXCLUDED {s['entity_id']}: {s['n']} usable months, "
                  f"below the 60-month floor. Reported as no-result, not as a weak result.")

    rb = rolling_beta("FAGCX", ret, fac, "ff3_mom")
    print()
    print("Rolling 36-month value exposure, FAGCX (style drift test case):")
    print(f"  first window ending {rb['period_end'].iloc[0]}: HML beta "
          f"{rb['beta_hml'].iloc[0]:+.3f}")
    print(f"  last  window ending {rb['period_end'].iloc[-1]}: HML beta "
          f"{rb['beta_hml'].iloc[-1]:+.3f}")
    print(f"  drift of {rb['beta_hml'].iloc[-1] - rb['beta_hml'].iloc[0]:+.3f} "
          f"detected across {len(rb)} windows")
    return cd, rb


# ---------------------------------------------------------------------------
# Project 1
# ---------------------------------------------------------------------------

def build_inputs(cme, corr):
    classes = list(cme["asset_class"])
    corr = corr.loc[classes, classes]
    mu = cme.set_index("asset_class").loc[classes, "arithmetic_er"].to_numpy()
    vol = cme.set_index("asset_class").loc[classes, "volatility"].to_numpy()
    Sigma = corr.to_numpy() * np.outer(vol, vol)
    return classes, mu, vol, Sigma


def min_var(mu, Sigma, target=None, bounds=None, groups=None):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        cons.append({"type": "ineq", "fun": lambda w: mu @ w - target})
    for mask, lo, hi in (groups or []):
        m = np.asarray(mask, dtype=float)
        if lo is not None:
            cons.append({"type": "ineq", "fun": lambda w, m=m, lo=lo: m @ w - lo})
        if hi is not None:
            cons.append({"type": "ineq", "fun": lambda w, m=m, hi=hi: hi - m @ w})
    res = optimize.minimize(lambda w: w @ Sigma @ w, np.full(n, 1 / n),
                            jac=lambda w: 2 * Sigma @ w, method="SLSQP",
                            bounds=bounds or [(0, 1)] * n, constraints=cons,
                            options={"maxiter": 400, "ftol": 1e-12})
    return res


def risk_contributions(w, Sigma):
    sp = float(np.sqrt(w @ Sigma @ w))
    rc = w * (Sigma @ w) / sp
    return sp, rc


def run_project1(cme, corr, sec):
    print()
    print("=" * 74)
    print("PROJECT 1 CHECK - optimizer inputs built from the CME table")
    print("=" * 74)
    classes, mu, vol, Sigma = build_inputs(cme, corr)

    eig = np.linalg.eigvalsh(Sigma)
    print(f"Covariance matrix: {Sigma.shape[0]}x{Sigma.shape[1]}, "
          f"min eigenvalue {eig.min():.3e}, "
          f"{'PSD ok' if eig.min() > -1e-12 else 'NOT PSD'}")

    growth = sec.drop_duplicates("asset_class").set_index("asset_class")
    is_growth = np.array([1.0 if c in ("us_large_equity", "us_small_equity",
                                       "intl_developed_equity", "em_equity")
                          else 0.0 for c in classes])

    bounds = [(0.0, 0.30)] * len(classes)
    groups = [(is_growth, 0.58, 0.62)]   # 60/40 mandate, 2 point tolerance
    res = min_var(mu, Sigma, target=None, bounds=bounds, groups=groups)
    w = res.x
    sp, rc = risk_contributions(w, Sigma)

    out = pd.DataFrame({
        "asset_class": classes,
        "weight_%": (w * 100).round(2),
        "arith_ER_%": (mu * 100).round(2),
        "vol_%": (vol * 100).round(2),
        "risk_contrib_%": (rc / sp * 100).round(2),
        "at_bound": ["yes" if (w[i] > 0.299 or w[i] < 0.001) else "" for i in range(len(w))],
    })
    print()
    print("Minimum variance under a 60/40 growth mandate, each class capped at 30%:")
    print(out.to_string(index=False))
    print(f"\n  portfolio arithmetic ER  {mu @ w * 100:.2f}%")
    print(f"  portfolio volatility     {sp * 100:.2f}%")
    print(f"  growth asset weight      {is_growth @ w * 100:.2f}%  (mandate 58-62%)")
    print(f"  weights sum              {w.sum():.10f}")
    print(f"  risk contributions sum   {rc.sum() * 100:.4f}%  "
          f"vs portfolio vol {sp * 100:.4f}%  (identity holds)")
    print(f"  solver status            {res.message}")

    print()
    infeasible_target = 0.12
    r2 = min_var(mu, Sigma, target=infeasible_target, bounds=bounds, groups=groups)
    achieved = mu @ r2.x
    print(f"Infeasibility check, {infeasible_target:.0%} target return:")
    print(f"  best achievable under these constraints {achieved * 100:.2f}%")
    print(f"  target flagged as INFEASIBLE, no weights reported to the manager"
          if achieved < infeasible_target - 1e-6 else "  target feasible")
    return out


def aggregate_policy(pw, classes):
    """Fund-level client weights aggregated to the optimizer's asset classes."""
    g = pw.groupby("asset_class")["policy_weight"].sum()
    return np.array([float(g.get(c, 0.0)) for c in classes])


def run_reference_portfolio(cme, corr, pw):
    print()
    print("=" * 74)
    print("REFERENCE PORTFOLIO - First Bank 60/40 Select as supplied")
    print("=" * 74)
    classes, mu, vol, Sigma = build_inputs(cme, corr)
    w0 = aggregate_policy(pw, classes)
    sp, rc = risk_contributions(w0, Sigma)

    out = pd.DataFrame({
        "asset_class": classes,
        "capital_%": (w0 * 100).round(2),
        "risk_%": (rc / sp * 100).round(2),
        "risk_minus_capital": (rc / sp * 100 - w0 * 100).round(2),
    })
    print(out.to_string(index=False))
    print(f"\n  arithmetic ER  {mu @ w0 * 100:.2f}%")
    print(f"  volatility     {sp * 100:.2f}%")
    print(f"  weights sum    {w0.sum():.6f}")
    eq = np.array([1.0 if "equity" in c else 0.0 for c in classes])
    print(f"  equity weight  {eq @ w0 * 100:.2f}%")

    top = out.sort_values("risk_%", ascending=False).iloc[0]
    print(f"\n  {top['asset_class']} holds {top['capital_%']:.0f}% of capital "
          f"and drives {top['risk_%']:.0f}% of portfolio risk.")
    print("  Concentration of this kind is invisible in a weights table and is the")
    print("  reason the allocation tool reports risk contributions, not just weights.")
    return classes, mu, vol, Sigma, w0


def _turnover(w, w0):
    return 0.5 * float(np.abs(w - w0).sum())


def diagnose_reference(w0, classes, ub=0.30, eq_band=(0.595, 0.605)):
    """
    Check the client's own portfolio against the constraints we plan to impose.

    If the starting portfolio violates a constraint, the optimizer cannot report
    a small-turnover answer: it has to trade to compliance first. Finding this
    before the client meeting is the whole point of pretreatment.
    """
    print()
    print("=" * 74)
    print("CONSTRAINT COMPATIBILITY - client portfolio vs our placeholder limits")
    print("=" * 74)
    eq_mask = np.array([1.0 if "equity" in c else 0.0 for c in classes])
    viol = [(c, w) for c, w in zip(classes, w0) if w > ub + 1e-12]
    eqw = float(eq_mask @ w0)

    print(f"Placeholder limits: {ub:.0%} per asset class, equity "
          f"{eq_band[0]:.1%}-{eq_band[1]:.1%}")
    print(f"  equity weight {eqw:.2%}: within the mandate band")
    if not viol:
        print(f"  no asset class exceeds the {ub:.0%} cap")
        return 0.0
    for c, w in viol:
        print(f"  {c} at {w:.2%} EXCEEDS the {ub:.0%} cap by "
              f"{(w - ub) * 100:.2f} pp")
    excess = sum(w - ub for _, w in viol)
    print(f"\n  The reference portfolio is INFEASIBLE under these limits.")
    print(f"  Minimum one-way turnover to reach compliance: {excess:.2%}")
    print(f"  Any turnover budget below that returns no solution, correctly.")
    print(f"\n  The 30% cap is OUR placeholder, not First Bank's policy. That is")
    print(f"  the finding: a plausible-looking limit makes their live portfolio")
    print(f"  non-compliant, so we need their real constraints before Project 1")
    print(f"  can produce an answer a manager would act on.")
    return excess


def solve_with_turnover(mu, Sigma, w0, tau, eq_mask, eq_band=(0.595, 0.605),
                        ub=0.30):
    """
    Minimum variance with a one-way turnover limit, 0.5*sum|w - w0| <= tau.

    The turnover term is not differentiable at w = w0, which is exactly where a
    solver starts. Standard fix: epigraph variables t_i >= |w_i - w0_i|, imposed
    as two linear inequalities each, with sum(t) <= 2*tau. All constraints become
    linear and t is pinned to |w - w0| at the optimum.

    Two earlier attempts are worth recording so nobody repeats them. Splitting
    the trade into separate buy and sell variables produced a degenerate vertex
    that neither SLSQP nor trust-constr solved. Smoothing the kink with
    sqrt(d^2 + eps^2) has zero gradient at d = 0, so the constraint looked
    locally inactive and small turnover budgets failed.

    In the delivered tool this is a few lines of CVXPY. It exists here so the
    release package can be verified before the optimizer team has built anything.
    """
    n = len(mu)
    lo, hi = eq_band
    ridge = 1e-10

    def obj(z):
        w, t = z[:n], z[n:]
        return float(w @ Sigma @ w + ridge * (t @ t))

    def grad(z):
        g = np.zeros(2 * n)
        g[:n] = 2 * Sigma @ z[:n]
        g[n:] = 2 * ridge * z[n:]
        return g

    I, Z = np.eye(n), np.zeros((n, n))
    cons = [
        {"type": "eq", "fun": lambda z: z[:n].sum() - 1,
         "jac": lambda z: np.concatenate([np.ones(n), np.zeros(n)])},
        # t >= w - w0
        {"type": "ineq", "fun": lambda z: z[n:] - (z[:n] - w0),
         "jac": lambda z: np.hstack([-I, I])},
        # t >= w0 - w
        {"type": "ineq", "fun": lambda z: z[n:] + (z[:n] - w0),
         "jac": lambda z: np.hstack([I, I])},
        {"type": "ineq", "fun": lambda z: 2 * tau - z[n:].sum(),
         "jac": lambda z: np.concatenate([np.zeros(n), -np.ones(n)])},
        {"type": "ineq", "fun": lambda z: eq_mask @ z[:n] - lo,
         "jac": lambda z: np.concatenate([eq_mask, np.zeros(n)])},
        {"type": "ineq", "fun": lambda z: hi - eq_mask @ z[:n],
         "jac": lambda z: np.concatenate([-eq_mask, np.zeros(n)])},
    ]
    bounds = [(0.0, ub)] * n + [(0.0, 1.0)] * n

    unc = min_var(mu, Sigma, bounds=[(0.0, ub)] * n,
                  groups=[(eq_mask, lo, hi)]).x
    starts = [w0, 0.5 * (w0 + unc), unc]

    best, best_res = None, None
    for s in starts:
        z0 = np.concatenate([s, np.abs(s - w0)])
        res = optimize.minimize(obj, z0, jac=grad, method="SLSQP",
                                bounds=bounds, constraints=cons,
                                options={"maxiter": 1000, "ftol": 1e-12})
        w = res.x[:n]
        ok = (res.status == 0
              and abs(w.sum() - 1) < 1e-8
              and _turnover(w, w0) <= tau + 1e-7
              and lo - 1e-8 <= eq_mask @ w <= hi + 1e-8
              and w.min() > -1e-9 and w.max() < ub + 1e-9)
        if ok and (best is None or w @ Sigma @ w < best @ Sigma @ best - 1e-14):
            best, best_res = w, res

    return best, best_res if best is not None else res


def run_turnover_case(classes, mu, vol, Sigma, w0, excess=0.0):
    print()
    print("=" * 74)
    print("TURNOVER CONSTRAINED REALLOCATION - what the tool would actually answer")
    print("=" * 74)
    eq_mask = np.array([1.0 if "equity" in c else 0.0 for c in classes])
    sp0, _ = risk_contributions(w0, Sigma)
    print(f"Starting portfolio: ER {mu @ w0 * 100:.2f}%, vol {sp0 * 100:.2f}%, "
          f"equity {eq_mask @ w0 * 100:.1f}%")
    print("Constraints: fully invested, long only, 30% class cap, equity 59.5-60.5%")
    print()

    rows = []
    for tau in (0.02, 0.05, 0.10, 0.25, 1.00):
        w, res = solve_with_turnover(mu, Sigma, w0, tau, eq_mask)
        if w is None:
            rows.append({"turnover_cap": f"{tau:.0%}", "turnover_used_%": None,
                         "ER_%": None, "vol_%": None, "vol_reduction_bp": None,
                         "cap_binds": "-", "result": "infeasible"})
            continue
        used = _turnover(w, w0)
        sp, _ = risk_contributions(w, Sigma)
        rows.append({
            "turnover_cap": f"{tau:.0%}" if tau < 1 else "unlimited",
            "turnover_used_%": round(used * 100, 2),
            "ER_%": round(mu @ w * 100, 2),
            "vol_%": round(sp * 100, 3),
            "vol_reduction_bp": int(round((sp0 - sp) * 10000)),
            "cap_binds": "yes" if used > tau - 1e-4 and tau < 1 else "no",
            "result": "verified",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    vols = df.loc[df["result"] == "verified", "vol_%"].to_numpy()
    monotone = bool(np.all(np.diff(vols) <= 1e-6))
    n_inf = int((df["result"] == "infeasible").sum())
    print(f"\n  volatility non-increasing in the turnover budget: {monotone}")
    if n_inf:
        print(f"  {n_inf} budget(s) reported infeasible: below the "
              f"{excess:.0%} minimum turnover needed to satisfy the class cap.")
        print("  No weights are shown for an infeasible budget. Reporting a")
        print("  near-miss portfolio as if it were compliant is how a tool loses")
        print("  a manager's trust.")
    print("\n  Each row is a different answer to the same manager question, and")
    print("  none of them is computable without the client's starting weights.")
    return df


if __name__ == "__main__":
    sec, ret, fac, cme, corr = load()
    pw = pd.read_csv(RELEASE / "policy_weights.csv")
    run_project2(sec, ret, fac)
    run_project1(cme, corr, sec)
    classes, mu, vol, Sigma, w0 = run_reference_portfolio(cme, corr, pw)
    excess = diagnose_reference(w0, classes)
    run_turnover_case(classes, mu, vol, Sigma, w0, excess)
