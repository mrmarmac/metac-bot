# Compare the actual model profiles prospectively

The weekly holdout asks whether at least three worldwide earthquakes will satisfy the USGS magnitude-6 catalog query in the next Monday-to-Monday UTC window. This independently authored question gives both profiles identical primary evidence, including eight prior weekly counts. Capturing before the event window also prevents the models from recalling the eventual outcome. This experiment measures binary count-event forecasting in one domain. [USGS catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)

With the encrypted journal initialized, export the same GitHub state credentials, funding selection, inference key, positive spend limits, and calibration configuration described in the README. Then run from the repository checkout, using a persistent directory. Choose the funding source whose terms permit this experiment. Metaculus describes benchmarking on custom questions as a permitted related use of donated inference credits. The experiment acquires its evidence directly from USGS. [Tournament credit terms](https://www.metaculus.com/notebooks/38928/)

```sh
uv run python forward_eval.py capture ~/futureeval-holdout --spend
uv run python forward_eval.py resolve ~/futureeval-holdout/YYYY-MM-DD.capture.json
uv run python forward_eval.py report ~/futureeval-holdout
```

Capture once per week, preferably at a consistent hour. Replace the date in the resolve command with the UTC Monday printed by capture. Resolve between 24 hours and seven days after the target week ends. The first saved catalog snapshot determines the outcome under the prespecified criteria. Subsequent reports use that snapshot offline.

Capture makes at most two model estimates, with rate-limit retries subject to the shared ledger. Both profiles reuse the same baseline response. Only the experimental profile adds scenario reasoning with the second model. Model failures exercise the production fallback. Each capture records both raw structured responses or failures, source observations, timing, per-call billing records, revision, configuration and hashes. A create-only local file and a claim in the shared ledger prevent repeat captures. A crash can leave an unsuccessful trial in the coverage denominator.

The report gives paired Brier and logarithmic losses, successful profile coverage, and summed member time, settled cost and outstanding reservations. Baseline cost and time are already included in experimental totals. Keep calibration settings fixed for each holdout series. For a comparison over many weeks, version the raw archive and report the number of paired windows with the scores. Choose subsequent settings using completed windows and evaluate them prospectively on fresh ones.

The automated tests exercise capture, claim, resolution, offline rescoring and cutoff enforcement with mocked calls. No paid forecast or prospective accuracy result is included in this checkout.
