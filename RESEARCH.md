# Forecasting design and evidence

## What earlier tournaments establish

Metaculus placed Panshul42 first in Q2 2025. The author combined historical reference-class research with a separate causal analysis, used several models and weighted their forecasts. In the same tournament a simple o3 baseline placed second. The result makes independent perspectives worth testing. It does not isolate the effect of the committee, scraping or model choice. [Season analysis](https://www.metaculus.com/notebooks/40456/q2-ai-benchmark-results/) · [Public implementation](https://github.com/Panshul42/Forecasting_Bot_Q2)

The organizer lists NoStream tenth in Spring 2026 and fourteenth in Summer. Its public implementation describes deadline-aware research and a median ensemble. Its author observed large errors when models agreed after seeing a shared briefing. The underlying repository evolves, so reproducing a past ranking requires checking out that season’s revision. Neither repository establishes a comparable dollar cost per successful forecast. [Organizer’s open-source bot listing](https://www.metaculus.com/notebooks/38928/#open-source-bots) · [NoStream implementation](https://github.com/No-Stream/nostreambot-metaculus-bot)

In Spring 2026, Metaculus found substantial uncertainty in the ordering of the top ten, a drop in team performance when increasingly weak members were added, and no clear search-provider winner in its controlled DeepSeek-R1 comparison. Model selection mattered in its comparisons, though survey associations across different bot architectures have confounders. These findings support keeping a cheap single-member comparison and measuring each additional member on the same questions. [Spring analysis](https://www.metaculus.com/notebooks/45373/spring-2026-futureeval-analysis/)

## Implementation choices

The runner discovers open questions, reads the bot account’s own state, preserves the question text and numerical scale, then passes an internal time budget to the inference layer. Research uses dated AskNews summaries when configured. Forecast estimators have no tools and cannot choose a submission destination. They must distinguish factual support from assumptions and match quoted support to retrieved evidence. Exact substring checks establish provenance. They do not verify whether a quoted claim is true or whether its interpretation is correct.

The live baseline uses one outside-view estimator. The experimental profile adds a different model with a causal-scenario prompt. Members see the same question and evidence but never see each other’s estimates. The aggregator averages probabilities or CDF ordinates across valid members. Identity calibration is the default. A five-quantile numeric forecast is inexpensive to request, at the cost of limited representation of multimodality and rare tails. Source availability, shared briefing errors, correlated model families and a constrained token budget all limit the ensemble.

Subscriptions, prize money and competition credits are different budget items. Direct comparisons should include successful coverage, dollar-equivalent inference usage, elapsed time and score over common resolved questions. Free-to-the-entrant inference can still exhaust donated credit. The provider price ceilings and durable reservations serve an operational purpose and do not establish an optimal spending level.

## Reproducing the historical comparison

Run `uv run --locked python evaluation.py --output evaluation_report.json`. The report records the lockfile, input data and evaluated code hashes alongside the Git base revision, calibration parameter, temporal windows, proper losses, coverage, wall time and actual API cost. Metric values are deterministic. Timings and source revision metadata depend on the checkout.

The fixture contains signed changes to the target federal funds range at all 32 scheduled meetings in 2022–25. Announcement dates come from the Board of Governors of the Federal Reserve System. For each date, the original statement is at `https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm`. The Board’s rate table gives effective dates, generally one day after the announcements. This fixture contains factual decision dates and changes. [Dated meeting archive](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) · [Historical rate changes](https://www.federalreserve.gov/monetarypolicy/openmarket.htm) · [Board reuse policy](https://www.federalreserve.gov/disclaimer.htm)

The first 16 meetings seed the reference class. Forecasts for every scheduled meeting in 2024–25 use only releases strictly before 00:00 UTC the day before that meeting. The outcome reaches only the scorer. Each decision produces four related forecasts covering hold versus change, cut/hold/raise, the signed change in basis points and its ordered 25-basis-point category. Sixteen decisions provide sixteen independent forecasting opportunities at most. Treating the 64 format-specific scores as independent observations would exaggerate precision.

The harness exercises the same question parser, forecast assembly, explanation validation, distribution construction, aggregation and output validation as the live profiles. It substitutes deterministic reference-class estimates for model calls and archived decision facts for online search. The first member uses all earlier meetings. The second uses the last four. Categorical estimates add one pseudocount per possible outcome. Numeric members fit a bounded Gaussian with a standard deviation floor of 12.5 basis points. The experimental profile takes the equal-weight mixture. These are modeling assumptions fixed in the code for a retrospective study. The experiment was not preregistered. The exercise measures the behavior of the recency and aggregation choices on historical monetary policy. It cannot estimate a live LLM’s tournament rank, search quality, cost or latency.

Binary scores are Brier and natural-log loss. Multiple choice uses multiclass log loss. Numeric CRPS integrates the squared error of the interpolated CDF in basis points. Discrete ranked probability score averages squared error across the interior category thresholds. Lower is better for each loss. The report includes counts and observed frequencies for five fixed binary reliability bins, omitting empty bins. FutureEval uses log-based spot peer scores against the tournament field. An exact tournament score requires competitors’ forecasts and platform scoring details. [Tournament scoring overview](https://www.metaculus.com/notebooks/38928/)

On the 16 held-out meetings, including the recent-history member lowered every average loss measured here. Ten of the 16 decisions were holds. The expanding-history estimator assigned only 35–46% on average across its occupied reliability bins, reflecting the rate increases in its early history. Both profiles generated all 64 format-specific forecasts at zero API cost.

| Held-out loss | Expanding history | Equal mix with last four |
| --- | ---: | ---: |
| Binary Brier | 0.2911 | 0.2591 |
| Binary log loss | 0.7772 | 0.7123 |
| Multiple-choice log loss | 1.4059 | 1.0856 |
| Numeric CRPS (basis points) | 20.60 | 12.53 |
| Discrete normalized RPS | 0.1007 | 0.0589 |

## Evaluating changes prospectively

Freeze the commit, profile, model version, price caps, calibration and evidence policy before collecting a new sample. Use self-authored non-prize questions with explicit resolution sources and forecast dates. At forecast time save the issued predictions, dated evidence snapshots, model identifiers, usage and latency privately. Score them after resolution. Historical model evaluations also require model training cutoffs preceding the simulated forecast dates. An article-date filter alone cannot remove memorized outcomes.

Compare paired losses on the same resolved events, report missing coverage and reasons separately, and estimate uncertainty by resampling events or temporal blocks. Fit calibration on an earlier development interval only. Freeze the fitted parameter and evaluate the later interval once. With only 16 held-out policy decisions, this report supplies no basis for selecting live calibration or claiming a competitive edge. Use current competition and data-access permissions before adding any new dataset. Do not review or adjust forecasts on open or upcoming prize questions. [Official rules](https://www.metaculus.com/notebooks/28751/futureeval-competition-rules/) · [Participation and testing resources](https://www.metaculus.com/futureeval/participate/)
