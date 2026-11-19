"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles, in percent. Python 3.13 standard library only; deterministic.

Model (structural, in earners space)
------------------------------------
A rate is earners / players_tracked. Two forces move it:

1. Organic earning: players who are already tracked keep earning trophies.
   Per trophy, the log of its earner count grows at a rate fitted from the
   history (median slope over intervals where players_tracked did not move,
   so refresh dilution cannot contaminate it), shrunk toward the tier
   median, and damped for long horizons (growth saturates as the easy
   trophies fill up).

2. Dilution by refresh: Exophase refreshes a game's counts on its own
   schedule, hours apart, and each refresh adds a batch of newly tracked
   players who hold far fewer trophies than the incumbents. Newcomers
   arrive holding trophy i at a fraction k_i of the incumbents' rate
   (k_i is larger for the common, early trophies: newcomers have played
   the opening, not the endgame). The forecast players count N(target)
   comes from the history's log-N trend when the history shows movement;
   when the history is perfectly flat (launch night: no refresh yet seen),
   a refresh prior fires instead: refreshes arrive about every 4.5 hours,
   the hazard rising the longer the flat stretch has run, and one refresh
   multiplies the tracked population by ~6 (the launch-night catch-up
   jump). The expected dilution over the horizon follows from that.

The median is the earners-space forecast above; the spread (logit units)
combines a random-walk term growing with sqrt(horizon), the fit residual,
and a refresh-uncertainty term that is large exactly when the refresh
prior is doing the work (short, flat-history horizons) and small once
the history itself shows the refresh rhythm.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (priors; see module docstring) ---
REFRESH_EVERY_H = 4.5      # typical gap between Exophase refreshes, hours
REFRESH_LOG_JUMP = 1.8     # ln(6.06): launch-night catch-up refresh size
NEWCOMER_HOLD = 0.08       # k_i scale: newcomers hold ~8% of incumbents' rate
K_POWER = 1.5              # k_i grows with (rate / max rate) ** K_POWER
SIGMA_RW = 0.5             # logit random-walk sigma per sqrt(day)
GROWTH_DAMP_DAYS = 3.0     # organic slope halves over ~this horizon
N_TREND_DAMP_DAYS = 2.0    # players-tracked growth saturates over ~this long
SIGMA_REF_PRIOR = 1.10     # refresh-uncertainty scale when prior is active
SIGMA_REF_FITTED = 0.30    # refresh-uncertainty scale once history shows it
SIGMA_FIT_FLOOR = 0.15
SIGMA_FIT_DEFAULT = 0.30
SIGMA_FIT_CAP = 0.80
SIGMA_MIN = 0.25
SIGMA_MAX = 2.0


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _round4(x):
    """Round to 4 decimals, half up; whole numbers stay whole."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def _ols_resid_std(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    slope = _ols_slope(xs, ys)
    mx = sum(xs) / n
    my = sum(ys) / n
    icept = my - slope * mx
    ss = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(ss / max(1, n - 2))


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = horizon_h / 24.0

    if not history:
        return {"trophies": [
            {"id": t["id"], "q": [0.1, 0.5, 1.0, 2.0, 5.0]} for t in trophies]}

    # --- history in hours since the first read ---
    t0 = _parse_time(history[0]["at"])
    times = [(_parse_time(h["at"]) - t0).total_seconds() / 3600.0 for h in history]
    players = [float(h["players_tracked"]) for h in history]
    n_last = players[-1]
    n_first = players[0]

    # last non-null rate per trophy, and its earner count
    last_rate = []
    for i in range(len(trophies)):
        r = None
        for h in reversed(history):
            if h["rates"][i] is not None:
                r = float(h["rates"][i])
                break
        last_rate.append(0.0 if r is None else r)
    last_earn = [max(r / 100.0 * n_last, 0.0) for r in last_rate]

    # --- 1. organic earner growth, per trophy ---
    # Interval slopes of log earners over stretches where N did not move.
    slopes = []
    resid_stds = []
    for i in range(len(trophies)):
        iv = []
        xs, ys = [], []
        for j in range(1, len(history)):
            if players[j] == players[j - 1]:
                r_prev = history[j - 1]["rates"][i]
                r_now = history[j]["rates"][i]
                if r_prev is None or r_now is None:
                    continue
                e_prev = max(float(r_prev) / 100.0 * players[j - 1], 0.5)
                e_now = max(float(r_now) / 100.0 * players[j], 0.5)
                dt = (times[j] - times[j - 1]) / 24.0
                if dt > 0:
                    iv.append((math.log(e_now) - math.log(e_prev)) / dt)
            xs.append(times[j] / 24.0)
            ys.append(math.log(max(
                (history[j]["rates"][i] or 0.0) / 100.0 * players[j], 0.5)))
        slope = _median(iv) if iv else 0.0
        slopes.append(min(max(slope, -0.2), 2.0))
        rs = _ols_resid_std(xs, ys) if len(xs) >= 3 else None
        resid_stds.append(rs)

    # shrink each slope toward its tier's median slope
    tier_slopes = {}
    for i, t in enumerate(trophies):
        tier_slopes.setdefault(t["tier"], []).append(slopes[i])
    tier_med = {k: _median(v) for k, v in tier_slopes.items()}
    growth = []
    for i, t in enumerate(trophies):
        g = 0.5 * slopes[i] + 0.5 * tier_med[t["tier"]]
        growth.append(min(max(g, -0.2), 2.0))

    # --- 2. players tracked at the target ---
    moved = any(players[j] != players[j - 1] for j in range(1, len(players)))
    if moved:
        # fit log N on the last 24 h of reads, extrapolate with saturation
        cut = times[-1] - 24.0
        xs = [t / 24.0 for t in times if t >= cut]
        ys = [math.log(p) for t, p in zip(times, players) if t >= cut]
        g_n = _ols_slope(xs, ys)  # log players per day
        g_n = min(max(g_n, 0.0), 3.0)
        dlogn = g_n * N_TREND_DAMP_DAYS * (1.0 - math.exp(-horizon_d / N_TREND_DAMP_DAYS))
        # Rail: the tracked population cannot plausibly grow more than ~12x
        # over one horizon once it is already days into launch; a short
        # window's slope must not compound into an absurd N(target).
        dlogn = min(max(dlogn, 0.0), 2.5)
        sigma_ref = SIGMA_REF_FITTED
        lam = None
    else:
        # flat history: refresh prior. The longer the flat stretch (first
        # read to the checkpoint), the more "due" a refresh is; expected
        # refreshes over the horizon:
        gap_h = (_parse_time(case["at"]) - t0).total_seconds() / 3600.0
        lam = (horizon_h / REFRESH_EVERY_H) * (1.0 + gap_h / REFRESH_EVERY_H)
        dlogn = min(lam * REFRESH_LOG_JUMP, 3.0)
        sigma_ref = SIGMA_REF_PRIOR * (1.0 - math.exp(-lam)) if lam else 0.0
    n_target = n_last * math.exp(dlogn)
    newcomers = max(n_target - n_last, 0.0)

    # --- 3. median forecast in earners space ---
    r_ref = max([r for r in last_rate if r < 99.99], default=1.0) or 1.0
    damp = 1.0 / (1.0 + horizon_d / GROWTH_DAMP_DAYS)
    rows = []
    for i, t in enumerate(trophies):
        r = last_rate[i]
        if r >= 99.99:
            q = [100, 100, 100, 100, 100]
            rows.append({"id": t["id"], "q": q})
            continue
        e0 = max(last_earn[i], 0.5)
        e_grown = e0 * math.exp(growth[i] * horizon_d * damp)
        k_i = min(NEWCOMER_HOLD * (max(r, 0.01) / r_ref) ** K_POWER, 0.5)
        e_new = k_i * (r / 100.0) * newcomers
        med = 100.0 * (e_grown + e_new) / n_target
        med = min(max(med, 0.01), 99.99)

        # --- 4. spread ---
        rs = resid_stds[i]
        sigma_fit = SIGMA_FIT_DEFAULT if rs is None else min(
            max(rs, SIGMA_FIT_FLOOR), SIGMA_FIT_CAP)
        sigma = math.sqrt((SIGMA_RW * math.sqrt(horizon_d)) ** 2
                          + sigma_fit ** 2 + sigma_ref ** 2)
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        lg = _logit(med)
        q = [_round4(min(max(_inv_logit(lg + z * sigma), 0.01), 100.0))
             for z in Z]
        # enforce non-decreasing after rounding
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": t["id"], "q": q})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
