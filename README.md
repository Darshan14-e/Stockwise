# From forecast to stock decision: how much inventory does a better forecast save?

A retail demand-forecasting model is only useful if it changes how much stock you order.
This project closes that gap. It forecasts demand over each supplier lead time, turns the forecast into
an order-up-to level, and then **simulates the inventory on days the model has never seen** to measure
what the forecast is worth.

> **Read this first.** On real Walmart M5 data the ML policies need less stock than a plain moving-average baseline
> at the same fill rate in all four runs I made (three stores, plus a different random draw of lead times):
> **about 2-10% less**, with the ML quantile policy the steadier of the two. That is a modest gain. On my synthetic data
> the gap is about 25%, but part of it is built into my own data generator. My first real-data run was much worse (the
> ML policies needed far MORE stock); the section "A failure worth knowing about" explains what fixed it.

## The business question

A store has to order stock days before customers buy it. Too little and sales are lost; too much and cash
is tied up in the stockroom. **How much stock is needed to reach a given service level, and can a better
forecast get there with less?**

## What it does

1. **Forecast** the units each product will sell during its lead time (the days between ordering and delivery).
   The model sees only past sales plus the planned promotion calendar. Two versions: a mean forecast and
   upper-quantile forecasts (50% to 99%).
2. **Decide**: turn forecasts into an order-up-to level under three policies:
   - **Baseline**: 28-day average x lead time + normal-distribution safety stock (the textbook method).
   - **ML + error buffer**: ML mean forecast + safety stock sized from that model's own validation errors.
   - **ML quantile**: use the ML quantile forecast directly as the stock target.
3. **Simulate** 198 unseen days of ordering, deliveries, sales and lost sales for every product and policy.
4. **Compare** the stock each policy needs to reach the *same* fill rate (share of demanded units actually sold).
   Comparing at equal fill rate is the fair test: a policy that holds more stock will of course sell more.
5. **Stress-test** the plan with late suppliers and a sudden demand jump.

## Results on synthetic data (60 products, 900 days, last 198 days unseen)

### 1. The same service level with about a quarter less stock

![Stock needed at each service level](results/fig_tradeoff.png)

| Fill rate | Baseline stock (₹ lakh) | ML + error buffer | ML quantile |
|---|---|---|---|
| 93.0% | 5.07 | 3.80 (-25%) | 3.84 (-24%) |
| 95.0% | 6.24 | 4.53 (-27%) | 4.57 (-27%) |
| 97.0% | 8.11 | 5.84 (-28%) | 5.82 (-28%) |
| 98.0% | 9.56 | 6.87 (-28%) | 6.84 (-29%) |

Percentages compare with the baseline; negative means less stock. Across 5 other random synthetic datasets the
saving of the ML quantile policy at a 95% fill rate ranged from **18% to 33%**, so the single-run figure is one
draw, not a promise.

### 2. All three policies set to a 95% target

| Policy | Fill rate | Avg stock (₹ lakh) | Holding cost (₹ lakh) | Lost margin (₹ lakh) | Total cost (₹ lakh) |
|---|---|---|---|---|---|
| Baseline | 96.2% | 7.12 | 0.90 | 9.61 | 10.51 |
| ML + error buffer | 97.9% | 6.68 | 0.84 | 5.43 | 6.27 |
| ML quantile | 97.7% | 6.40 | 0.81 | 5.86 | 6.67 |

Total cost = holding cost (25% of unit cost per year) + margin lost on missed sales. The policies end up at
different fill rates, so this table mixes better forecasting with higher service. The equal-fill table above
is the cleaner comparison.

### 3. The ML forecasts are better calibrated

![Calibration](results/fig_calibration.png)

| | Baseline | ML + error buffer | ML quantile |
|---|---|---|---|
| Error of the average forecast (WAPE, lower is better) | 23.0% | 18.9% | n/a |
| Pinball loss at the 95% quantile (lower is better) | 1.81 | 1.21 | 1.21 |
| Share of demand covered when aiming for 95% | 90.3% | 93.9% | 93.2% |
| Share of demand covered when aiming for 99% | 95.4% | 97.7% | 98.0% |

The baseline assumes demand is normally distributed and ignores promotions, so it under-protects.
The ML policies come closer to their targets but still fall slightly short.

### 4. What if the plan is wrong?

![Stress tests](results/fig_sensitivity.png)

| Total cost (₹ lakh), 95% target | Baseline | ML + error buffer | ML quantile |
|---|---|---|---|
| As planned | 10.51 (fill 96.2%) | 6.27 (fill 97.9%) | 6.67 (fill 97.7%) |
| Suppliers 2 days slower than planned | 40.34 (fill 84.7%) | 37.86 (fill 85.8%) | 39.95 (fill 85.1%) |
| Demand jumps 25% after test day 30 | 15.00 (fill 95.5%) | 13.82 (fill 95.9%) | 12.05 (fill 96.4%) |

**The main lesson:** a better forecast helps a lot when the plan holds, but late deliveries hurt every policy
(fill rate falls to about 85% for all of them). Forecasting cannot fix supplier unreliability; that needs
lead-time variability built into the safety stock.

## Results on real M5 data (Walmart): a modest, consistent win

Setup: store CA_1, the 60 best-selling items (chosen using training days only), 900 days, last 198 days unseen.
Unit cost (70% of shelf price) and lead times (2-10 days, seeded) are assumptions, because M5 has neither.
There is no promotion feature on M5 (M5 has realised prices, not a planned promotion calendar).

### Stock needed for the same fill rate

| Fill rate | Baseline stock (US$ thousand) | ML + error buffer | ML quantile |
|---|---|---|---|
| 93.0% | 2.81 | 2.71 (-4%) | 2.68 (-4%) |
| 95.0% | 3.38 | 3.21 (-5%) | 3.22 (-5%) |
| 97.0% | 4.29 | 4.18 (-3%) | 4.12 (-4%) |

Negative means less stock than the baseline. Charts: `results_m5/fig_tradeoff.png`, `fig_calibration.png`, `fig_sensitivity.png`.

### Forecast quality on unseen days

| | Baseline | ML + error buffer | ML quantile |
|---|---|---|---|
| Error of the average forecast (WAPE, lower is better) | 27.4% | 24.3% | n/a |
| Pinball loss at the 95% quantile (lower is better) | 5.40 | 4.04 | 4.25 |
| Share of demand covered when aiming for 95% | 85.9% | 94.2% | 93.1% |
| Share of demand covered when aiming for 99% | 91.7% | 97.0% | 96.7% |

### Stress tests (95% target, total cost in US$ thousand)

| Scenario | Baseline | ML + error buffer | ML quantile |
|---|---|---|---|
| As planned | 5.50 (fill 94.5%) | 2.68 (fill 97.7%) | 3.05 (fill 97.2%) |
| Suppliers 2 days slower than planned | 14.09 (fill 83.4%) | 10.09 (fill 88.4%) | 11.58 (fill 86.4%) |
| Demand jumps 25% after test day 30 | 7.38 (fill 93.8%) | 4.68 (fill 96.2%) | 4.48 (fill 96.3%) |

### Does it hold up? Four runs

Same code, 60 best-selling items per run, 900 days, last 198 unseen. Saving = stock saved versus the baseline at the
same fill rate (positive is better; the numbers come straight from each run's `headline.csv`).

| Run | Fill rate | ML + error buffer | ML quantile |
|---|---|---|---|
| CA_1 (main run) | 93% / 95% / 97% | 3.6% / 5.1% / 2.7% | 4.5% / 4.8% / 4.0% |
| TX_1 | 93% / 95% / 97% | 4.2% / 2.4% / 2.1% | 10.4% / 9.3% / 7.8% |
| WI_1 | 93% / 95% / 97% | 4.0% / 1.6% / 0.9% | 8.7% / 9.0% / 9.4% |
| CA_1, other lead-time draw (`--seed 8`) | 93% / 95% / 97% | 4.3% / 6.0% / 3.4% | 4.2% / 4.6% / 2.5% |

- Both ML policies beat the baseline in every run. The ML quantile saving ranges from 2.5% to 10.4%; the error-buffer
  saving from 0.9% to 6.0% (at a 98% fill rate in WI_1 it is about zero).
- The ML quantile policy is the steadier one: it is ahead by 8-10% in TX_1 and WI_1, while the error-buffer policy fades
  at high fill rates there. In the CA_1 runs the two are close and the error-buffer policy is sometimes slightly ahead.
- The same pattern shows in the forecast tables: in every run the ML quantile covers 93-94% of demand when aiming for
  95%, the baseline only 84-87%.
- The stress tests agree: with suppliers 2 days slower or demand 25% higher, both ML policies reach a higher fill rate
  at a lower total cost than the baseline in all four runs, yet fill still falls to about 82-88% when suppliers are late.

### What this says

- **The forecast got much better; the stock saving is small.** The pinball loss at the 95% level fell by about 25%, but
  the stock needed for the same fill rate fell only 3-5% on the main run. Better forecasts help, but on this data they are worth far
  less than the synthetic results suggest.
- **The baseline's real weakness is coverage.** Aiming for 95%, it covers only 85.9% of demand; the ML policies cover
  about 93-94%. At a given target the ML policies hold more stock because they are actually protecting what they claim.
  That is why the "95% target" total-cost table looks lopsided: the policies end at different fill rates. Compare the
  equal-fill table above instead.
- **Which ML policy?** On the main run they are equal; across the four runs the quantile policy is steadier. So the
  simpler error-buffer policy is a fair choice for one store, but the quantile policy is safer across stores.
- **Forecasting does not fix late suppliers.** With suppliers 2 days slower, every policy falls to 83-88% fill.
- **How far to trust the 2-10%:** it is three stores in three states, 60 items each, with assumed unit costs and random
  lead times, and no confidence intervals. The direction is consistent; the exact size is not pinned down. Stores in the
  other three states (CA_2-4, TX_2-3, WI_2-3) and a bootstrap over items would tighten it.

## A failure worth knowing about

My first run on real M5 data went badly: the ML policies held far more stock than the baseline, and the
"stock needed at the same fill rate" table came out empty. The cause was in how the models were trained. The
training target is *demand over the lead time divided by the recent daily average*. After a stretch of zero
sales that average is tiny, so the ratio can reach the hundreds, and a handful of such rows dragged the models.

Two fixes:

- **Cap the training target** at 4x the recent run-rate (`--target-cap`, default 4; `0` switches it off).
  On data with long zero-sales stretches, switching the cap off made the ML policies hold up to 2-3x the
  baseline's stock; with the cap on they used less. On the synthetic data above the cap changes nothing.
- **Make the comparison fair**: each policy is swept over wide enough service settings that their fill-rate
  ranges overlap, and the quantile forecasts are sorted so a higher service level never gives a lower target.

Two further fixes after the first M5 run:

- **No look-ahead in setup.** The M5 loader now picks the 60 items and their prices from training days only (before,
  "best sellers" were ranked over all days, including the test period).
- **No invented promotion calendar.** M5 has realised prices, not a planned promotion schedule, so on M5 the
  promotion feature is switched off instead of being derived from price cuts.

Every run prints a **data health** block (largest training ratio, share of zero-sales rows, forecast bias) so you
can see whether this applies to your data. Run once with `--target-cap 0` and once with the default to see the
effect on yours.

## Honest limitations

- **The synthetic data is mine.** The generator includes promotions that the ML model is told about and the
  baseline is not, so part of the gap is built in. Expect smaller gains on real data.
- **The baseline is deliberately simple.** A stronger baseline (seasonal or exponential-smoothing forecasts)
  would narrow the gap.
- **The two ML policies are about equal** on the synthetic data, so the simpler one (mean forecast plus an
  error buffer) is as good there. On the harder test data I tried, the quantile policy was the more reliable one.
- **The cap limits the model**: it cannot plan for demand above 4x the recent run-rate.
- **Simplifications:** fixed lead times, no minimum order quantities, orders placed daily (so a fixed cost per order
  is not modelled), lost sales are not backordered, and cost counts only holding and lost margin.
- **Costs and lead times are assumptions** (unit cost, 25% holding rate, 2-10 day lead times), not measured values.
  M5 has no cost or lead-time data, so on M5 unit cost is 70% of the shelf price and lead times are random (seeded).

## How it avoids cheating

- Time-based split: train, then validation, then test. Test days are never used for fitting.
- Features at decision day *t* use only data up to day *t*. A test scrambles all future data and checks that
  the features do not change.
- The simulator is unit-tested on cases with known answers (no stock-outs when stock covers lead-time demand,
  zero sales when the target is zero), and other tests cover quantile ordering, the target cap and empty comparisons.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py            # synthetic data, about 10 seconds; writes tables and charts to results/
python summarize.py results       # prints the results as Markdown tables
python tests/test_pipeline.py     # sanity checks
```

### Run it on real data

1. Download *M5 Forecasting - Accuracy* from Kaggle and unzip it into a folder **outside** this project.
2. Run:
   ```bash
   python run_pipeline.py --data m5 --m5-dir ../m5-data --out results_m5
   # other stores / seeds: add --store TX_1 --seed 8 --out results_m5_tx1
   python summarize.py results_m5
   ```

The loader picks one store (default `CA_1`), keeps its 60 best-selling items (ranked on training days only) and uses no promotion feature.
Do not commit the M5 files to GitHub.

## Project layout

```
run_pipeline.py        runs everything
summarize.py           turns a results folder into README tables
src/data.py            synthetic generator + M5 loader
src/forecast.py        features, LightGBM mean/quantile models, baseline, metrics
src/inventory.py       daily inventory simulation
src/report.py          charts
tests/test_pipeline.py sanity checks
results/               tables and charts from the latest synthetic run
```

## Ideas to extend it

- Add lead-time variability to the safety stock and re-run the "suppliers slower" test.
- Add a stronger baseline (exponential smoothing) and a weekly review cycle with minimum order quantities.
- Segment by ABC class and set different service levels for fast and slow movers.
- Build a Power BI page on `policy_curves.csv` so a non-technical reader can pick a service level and see the stock needed.
