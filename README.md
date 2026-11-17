# GTA VI launch-day Twitch viewers forecast

Entry program for make.fun's `gta6-twitch-ccu-forecast-01`: for each
case file in the input directory, forecast five quantiles (0.05, 0.25,
0.5, 0.75, 0.95) of the Twitch category's concurrent viewers for hours
2..24 of launch day, given everything recorded up to the end of hour 1.

Run it the way the grader does (Python 3.13, standard library only, no
network):

```
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Without the environment variables it reads `/input` and writes
`/output`. It is deterministic: two runs of a case produce byte-identical
output, and a case takes well under a second.

## The model

Everything is learned from the case's own `comparables` (past launches
with hourly viewers around their release days); nothing is hard-coded
per game.

1. For each comparable, bucket its hourly points by whole-hour offset
   from its `day_start`, log-interpolate gaps, and form the launch-day
   log-ratios `L_h = ln(v_h / v_1)` for hours 2..24.
2. The forecast for hour `h` is `p1 * exp(mu_h + z_tau * sd_h)`, where
   `mu_h`, `sd_h` are the weighted mean and standard deviation of the
   pool's `L_h`, and the `z_tau` are the empirical quantiles of the
   pool's standardized residuals pooled over all hours — so the spread's
   shape comes from the data, not from a normal assumption.
3. Pool weights are Gaussian in log hour-1 viewers (bandwidth 1.2):
   launches of similar scale dominate the pool.
4. Unlock pattern, parsed from the case's text (a comparable's `unlock`
   field; the target's `notes`):
   - *Rolling midnight* target: global-unlock and late-surge
     comparables leave the pool — their day shapes cannot occur.
   - *Global unlock* target with a parsed unlock hour `u`: copies of
     global/surge curves, time-shifted so their unlock hour (parsed,
     or the hour the curve reaches 60% of its log climb to peak) lands
     on `u` (and `u ± 3` at half weight), are mixed in at 35% of the
     pool weight.
   - *Cold* target (hour-1 print under 20,000, pattern unknown): the
     same shifted copies are added over a grid of candidate unlock
     hours {5, 9, 13, 17, 21}, integrating unlock timing out instead of
     guessing it.
5. Fallbacks: too few usable comparables, or any data problem, falls
   back to the house baseline's formula (flat at hour 1, normal spread
   `0.15 * sqrt(h - 1)`). Every case file — even a malformed one —
   produces a valid output file; quantiles are clamped to (0, 1e9] and
   forced non-decreasing.

## Validation

`dev/validate.py` drives the shipped `main.py` itself:

- **Leave-one-out cross-validation** over the 37 usable comparables
  pooled from the practice inputs: each launch in turn is the
  pseudo-target (its launch-day hour-1 as the known print, the other
  launches as its comparables), scored with the contest's mean pinball
  loss on the log scale against its raw observed hours 2..24.
  - House baseline formula: **LOO 0.9297**
  - This model: **LOO 0.5197** (rolling-midnight pseudo-targets: 0.33)
- **End-to-end** on the practice cases p00..p03: outputs pass the
  contest's shape checks, reruns are byte-identical, runtime is
  0.17–0.20 s per case (limit 8 s).
- `dev/edge_test.py`: hour-1 print 0 / null / missing, no comparables,
  sparse or malformed comparables, malformed case files — all still
  yield valid forecasts.

Why the LOO number is not lower: the pooled comparables include
"cold" launches whose hour-1 print is nearly zero because their single
global unlock falls later in the UTC day (e.g. hour-1 prints of 164,
414, 419 viewers against peaks of 150k–460k). For those, hour 1 carries
almost no information about the day's scale, and even the aligned,
mixture-based forecasts score 1.2–1.8. The contest's actual targets are
the opposite shape — the four practice launches and GTA VI itself are
rolling-midnight console launches with substantial hour-1 prints — and
on rolling pseudo-targets the model scores 0.33, with the largest
blockbuster launches (the final target's nearest neighbours by scale)
at 0.26–0.39. The house baseline scores ≈0.387 on the practice cases
per the kit's rules; the same formula scores 0.9297 under this pooled
LOO protocol, which is the like-for-like comparison.

Development history and the experiments behind these choices are in
`dev/` (`common.py`, `loo.py`, `model2.py` … `model5.py`, `diag.py`,
`explore.py`).

## Layout

- `main.py` — the entry program (the only file the grader runs).
- `LICENSE` — MIT, from the kit's starter.
- `dev/` — validation and the experiments that tuned the model.
