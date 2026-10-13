# Assignment 1 Report

> **✍️ = Manuela writes this part in her own words.** Everything else is the
> factual record of what the pipeline does and what it produced (numbers come
> from `artifacts/metrics.json`). Delete every ✍️ line once you've written your
> answer in its place.

- **Name**: ✍️
- **Student ID**: ✍️
- **Email**: mcarreira.ieu2022@student.ie.edu
- **Group**: ✍️ [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset

**Source.** Inside Airbnb (insideairbnb.com), "Detailed Listings" file for
Madrid, scraped in June–July 2026. Committed as `data/listings.csv.gz`
(10 MB) and loaded directly by `train.py`, so every cleaning step is
reproducible from the raw file.

**What a row is.** One Airbnb listing as it appeared on the scrape date.

**Size.** 22,708 listings × 90 columns. After scoping to entire homes:
15,899 listings, of which 13,936 have a price and are used for modelling
(1,963 have no price and cannot be used to train or evaluate).

**Target.** Nightly price in euros (`price`, stored as text such as
`"$1,234.00"`), modelled as `log(price)`. Median price €144; 90% of
listings between €71 and €421; maximum €36,883.

**Why this dataset.**
✍️ *Why Madrid Airbnb? What makes it interesting to you / why does feature
engineering matter here (messy amenities text, coordinates, regulation)?*

## Business / real-life framing

**The decision the model supports.** An overpricing detector for
entire-home listings in Madrid. The model estimates a listing's *fair*
nightly price from what the listing is (location, size, amenities, quality,
rules). A listing is flagged when its actual price is more than X% above
that fair price.

✍️ *Who is the user (a platform trust team? the city council? a consumer
association?) and what do they do with a flagged listing (manual review,
warning to host, ...)? Why does it matter?*

**How the framing drove the pipeline.**

| Decision | What was done | ✍️ Why (your reasoning) |
|---|---|---|
| Scope | Entire homes only (rooms excluded) | ✍️ |
| Target definition | `log(price)` instead of `price` | ✍️ *hint: the business rule is a percentage ("X% above fair")* |
| Split strategy | Grouped by `host_id`: 70/15/15 → 9,303 train / 2,183 validation / 2,450 test listings; no host appears in two sets | ✍️ *hint: one host has 373 listings* |
| No time-based split | Single snapshot, no history over time | ✍️ |
| Decision threshold | Flag if actual > fair × (1 + X%); X is a slider in the dashboard (default 50%) together with a cost per manual review | ✍️ *how would you pick X? trade-off between review cost and missed overpricing* |
| Evaluation metrics | RMSE and R² on log-price (what the model optimises); MAE in € and median % error (what the business reads) | ✍️ |

## Data preparation & feature engineering

**Cleaning and data-quality decisions**

- Price parsed from text (`"$1,234.00"` → 1234.0). Rows without a price dropped.
- **Extreme prices kept** (e.g. a 1-bedroom listed at €15,000/night).
  ✍️ *Why did you decide to keep them, and what is the cost of that choice?*
- 12 host columns are 100% empty in this scrape (response rate, host since,
  instant bookable, ...) and could not be used.
- `bathrooms` is missing for ~9% of rows; filled from the text column
  `bathrooms_text` ("1.5 baths").
- Remaining missing numeric values filled with the **training-set median**.
  Missing review scores happen exactly when a listing has no reviews, so the
  flag `has_reviews` tells the model the score was imputed.
- All imputation, scaling and one-hot encoding are **fitted on the training set
  only** and then applied to validation and test.

**Engineered features** (29 numeric + 2 categorical → 54 columns after one-hot)

| Group | Features | How |
|---|---|---|
| Location | `log_dist_to_sol_km`, `district` (21 districts) | km to Puerta del Sol from lat/lon, then log; district one-hot |
| Size | `accommodates`, `bedrooms`, `beds`, `bathrooms` | bathrooms recovered from text |
| Property type | 5 common types + "Other" | rare types grouped |
| Amenities | `n_amenities` + 10 flags: air conditioning, dishwasher, elevator, balcony/terrace, pool, parking, washer, workspace, self check-in | keyword matching over 2,512 different amenity labels (e.g. "Air conditioning" and "AC – split type ductless system" are the same thing) |
| Rules / legal | `is_monthly_rental` (min. stay ≥ 32 nights), `log_minimum_nights`, `has_license` | 200 listings are monthly rentals (median €56 vs €145) |
| Host | `is_superhost`, `log_host_listings` | professional vs individual hosts |
| Reviews / quality | `has_reviews`, `log_n_reviews`, `reviews_per_month`, 4 review scores | |
| Demand | `availability_365`, `estimated_occupancy_l365d` | included by decision |
| **Excluded** | `estimated_revenue_l365d` | computed as price × occupancy → contains the target (leakage) |

✍️ *For 2–3 of these features, explain in your own words why you expected them
to matter for price (e.g. distance to Sol, monthly rentals, amenities).*
✍️ *Why did you decide to include availability and occupancy, given that an
overpriced flat tends to stay empty? What is the risk?*

**Did the feature engineering help?** Same Ridge model, same split, scored
on validation:

| Feature set | Validation RMSE (log) |
|---|---|
| Raw columns only (accommodates, bedrooms, beds, latitude, longitude) | 0.497 |
| Engineered features | 0.402 |

✍️ *Interpret this.*

## Modeling: three implementations, one model

**Model.** Linear regression with L2 regularisation (Ridge) on
`log(price)`. ✍️ *Why linear regression (target is continuous)?*

**The three implementations**

1. **scikit-learn** `Ridge(alpha)`: minimises Σ(y − ŷ)² + α·Σw².
2. **Manual PyTorch loop** (Session 5 pattern): tensors `w`, `b` with
   `requires_grad=True`, forward pass `X @ w + b`, loss = MSE + λ·Σw²,
   `loss.backward()`, update inside `torch.no_grad()`, then `.grad.zero_()`.
   Full-batch gradient descent, 3,000 epochs.
3. **Standard PyTorch**: `nn.Module` containing `nn.Linear`, `nn.MSELoss`,
   `torch.optim.Adam`, loop of `zero_grad → forward → backward → step`,
   3,000 epochs.

So that all three solve **the same problem**, PyTorch uses
λ = α / n_train (dividing Ridge's sum-of-squares objective by n gives the mean
squared error plus (α/n)·Σw²). The bias is not penalised in any of the three.

**Tuning (sparingly, on validation only)**

| Hyperparameter | Values tried | Chosen |
|---|---|---|
| Ridge α | 0.1, 1, 10, 100, 1000 → val RMSE 0.4020, 0.4020, 0.4022, 0.4031, 0.4076 | 0.1 (λ = 1.07e-5) |
| LR manual loop | 0.01, 0.05, 0.1 | 0.1 |
| LR Adam | 0.01, 0.05, 0.1 | 0.05 (0.01 under-fitted: val RMSE ≈ 0.50) |

**Results on the test set (2,450 listings, hosts never seen in training)**

| Model | RMSE (log) | R² (log) | MAE (€) | Median % error |
|---|---|---|---|---|
| Naive baseline (mean log-price → €156 for everyone) | 0.507 | −0.012 | 68.27 | 32.4% |
| scikit-learn Ridge | 0.382 | 0.425 | 51.47 | 22.1% |
| Manual PyTorch loop | 0.382 | 0.424 | 51.56 | 21.9% |
| Standard PyTorch (nn + optim) | 0.382 | 0.425 | 51.48 | 22.0% |

**Do the three agree?**

| Comparison | Max difference in predictions (log) | Max difference in a coefficient |
|---|---|---|
| Standard PyTorch vs scikit-learn | 0.017 | 0.013 |
| Manual PyTorch vs scikit-learn | 0.100 | 0.124 |

Observed facts about the gap:
- Standard PyTorch (Adam) and scikit-learn match closely.
- The manual loop's differences are concentrated in the **property-type
  dummies and the intercept**: e.g. "Entire serviced apartment" 0.161 (manual)
  vs 0.049 (sklearn), "Other" +0.027 vs −0.097, intercept 4.90 vs 5.00. Also
  in rare districts (Barajas 0.602 vs 0.658; Moratalaz 0.082 vs 0.118).
- ~90% of listings are "Entire rental unit", so that dummy and the intercept
  move almost together; the rare districts have very few listings, so their
  columns are almost always 0. Both are directions where plain gradient
  descent receives tiny gradients and moves slowly, while scikit-learn solves
  the problem exactly and Adam rescales each parameter's step.
- Despite this, test metrics are almost identical (RMSE 0.3824 vs 0.3822).

✍️ *Explain in your own words why the manual loop hadn't fully converged and
why Adam got closer. What would you change to make it converge (more epochs?
a different learning rate?)*

**Most important coefficients** (effect on log-price per 1 standard deviation,
or versus the reference category: district = Arganzuela, property type =
Entire condo):

| Feature | Coefficient (sklearn) |
|---|---|
| district_Barajas | +0.658 |
| district_Hortaleza | +0.318 |
| district_Salamanca | +0.269 |
| log_minimum_nights | −0.232 |
| district_Retiro | +0.215 |
| log_dist_to_sol_km | −0.172 |
| accommodates | +0.115 |
| bedrooms | +0.114 |
| bathrooms | +0.095 |
| is_monthly_rental | −0.080 |

✍️ *Interpret 2–3 of these. Does anything surprise you (e.g. Barajas, the
airport district, being the most expensive)?*

**What the detector finds.** At a 50% threshold, 282 of 2,450 test listings
(11.5%) are flagged; median excess €103/night. Several of the most-flagged
listings have event names in their titles ("Final Champions Pizarro",
"Only 4 champions!!", "*Bernabeu Home*", "Stadium transfer").
✍️ *What does that tell you about what the model can and cannot see?*

## Limitations & next steps

Facts to build on:
- R² of 0.42: the model explains less than half of the variation in log-price.
- Extreme prices kept in training (up to €36,883).
- Regularisation made almost no difference (all α within 0.006 val RMSE).
- The manual gradient-descent loop was not fully converged on rare categories.
- `has_pool` showed no raw price difference.
- Predictions are converted back with exp(log-price), which gives a "typical"
  (median-like) price rather than the average.
- Event pricing (football finals, concerts) is invisible to the features.
- One snapshot: no seasonality.

✍️ *Pick the 3–4 limitations you think matter most, and for each one say
concretely how you would fix it with more time or data.*

## Generative AI use disclosure

✍️ *Write this honestly in your own words. What happened in practice: Claude
(Anthropic) helped with environment setup, wrote the code for `train.py` and
`app.py` (data loading, feature-engineering code, the three training loops,
saving models, the Gradio dashboard) based on decisions you made (dataset,
overpricing framing, entire homes only, keeping extreme prices, including
availability/occupancy, keeping monthly rentals with a flag), and prepared the
factual parts and tables of this report. The reasoning in the ✍️ sections is
your own.*
