# GTA VI Trophy Rarity Forecaster

[![bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ca7-cok/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ca7-cok/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for [make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): given Exophase's recorded earn rates for every Grand Theft Auto VI PlayStation 5 trophy, it forecasts each trophy's rate at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads one case (`/input/<id>.in`, or `$MAKEFUN_INPUT_DIR`) and writes `/output/<id>.out` (`$MAKEFUN_OUTPUT_DIR` outside the sandbox). The model works in **earners space** — a rate is earners / players tracked — because two different forces move a rate:

1. **Organic earning.** Players already tracked keep earning trophies. Per trophy, the log of its earner count grows at a rate fitted from the history (median slope over intervals where `players_tracked` did not move, so refresh dilution cannot contaminate it), shrunk toward the tier median and damped for long horizons as the easy trophies fill up.
2. **Dilution by refresh.** Exophase refreshes a game's counts on its own schedule, hours apart, and each refresh adds a batch of newly tracked players who hold far fewer trophies than the incumbents. Newcomers arrive holding each trophy at a fraction of the incumbents' rate (larger for the common, early trophies). The forecast player count comes from the history's log-N trend when the history shows movement; when the history is perfectly flat (launch night — no refresh yet seen), a refresh prior fires instead: refreshes arrive about every 4.5 hours, the hazard rising the longer the flat stretch has run, and one refresh multiplies the tracked population by ~6 (the launch-night catch-up jump).

The spread (logit units) combines a random-walk term growing with √horizon, the fit residual, and a refresh-uncertainty term that is large exactly when the refresh prior is doing the work (short, flat-history horizons) and small once the history itself shows the refresh rhythm. The forecast is horizon-aware: a 90-minute practice horizon is dominated by refresh risk, while the 6,510-minute final horizon is dominated by a week of real earn-rate growth.

The program is deterministic: two runs on the same case write byte-identical output, in well under the 8-second cap.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

The kit publishes no practice answers (`grade-local` grades program/v1 kits on the server only), so practice answers were reconstructed from the aggregates the competition itself states (3,868 players tracked, 4,276 trophies earned at the 04:30Z refresh; every rate but the 100% one fell), under eight different distributions of the 103 new earners. The persistence baseline scores 1.39–1.66 pinball loss across those reconstructions (the competition states ~1.51); this forecaster scores **0.25–0.44**, beating the baseline on every reconstruction, both practice cases, with 90% coverage 0.93–1.00 (baseline: 0.00).

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
