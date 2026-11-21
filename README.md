# Autonomous FutureEval forecaster

A GitHub Actions bot for the Fall 2026 seasonal tournament (33121) and rolling MiniBench (minibench). It forecasts binary, multiple-choice, numeric and discrete questions, submits once per question and attaches a private explanation. A fresh installation starts paused, with zero inference budget. The baseline uses one estimator. An optional experimental profile adds a second model and averages their probability distributions.

## Run without credentials

Install Git and uv. Use Python 3.11–3.13. Run these commands in the checkout.

```sh
uv sync --locked
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python runner.py --mode dry-run
uv run --no-sync python evaluation.py
```

Dry-run rehearses research, two model responses and final payloads on synthetic questions with mocked transports. The tests also run the original no-framework starter, exercise submission and comment recovery, and enforce budget and format constraints. The historical harness writes `evaluation_report.json`. Its expanding-history and recent-history estimates use known-at-the-time Federal Reserve releases. See [the research report](RESEARCH.md), [measured results](evaluation_report.json) and the [prospective model comparison](REPLAY.md).

## Connect the bot

1. Create one Metaculus bot account, register it for the competition, accept the terms and obtain the **bot's** API token and numeric profile ID. Only the account holder can perform registration. Apply for donated OpenRouter credits if desired. AskNews provides separate seasonal credentials. [Participation](https://www.metaculus.com/futureeval/participate/) · [Access and credits](https://www.metaculus.com/notebooks/38928/) · [Rules](https://www.metaculus.com/notebooks/28751/futureeval-competition-rules/)
2. In GitHub, enable Actions and allow the FutureEval workflow to write repository contents. The runtime token is automatically scoped to this repository. Use **Settings → Secrets and variables → Actions** for the table below. Keep production paused during setup.
3. Initialize the encrypted state **once per installation**, using the steps below. Preserve the state branch and key across upgrades. An existing, initialized deployment can skip this step.
4. Choose a funding source, set both budget caps and configure the corresponding key. In **Actions → FutureEval → Run workflow**, select the default branch and `test-area`. This performs real API calls and private submissions exclusively on the official bot-testing-area feed. Check the resulting health artifact. A zero-question run confirms connectivity only.
5. After a successful testing-area run, set the repository variable `FUTUREEVAL_ENABLED=true`. Scheduled jobs start on minutes 7, 27 and 47 UTC. GitHub can delay or drop scheduled jobs. An active run processes up to three questions at once. The forecast deadline uses the platform close and spot-scoring times. Jobs reserve time for submission and skip questions when too little time remains.

| Setting | Where | Value or purpose |
|---|---|---|
| METACULUS_TOKEN | Secret | Bot account token |
| METACULUS_BOT_ID | Variable | Numeric bot account ID for own-history audits and private comments |
| STATE_KEY | Secret | Fernet key for the encrypted journal |
| FUTUREEVAL_ENABLED | Variable | `false` by default, `true` enables production |
| FORECAST_FUNDING | Variable | `disabled`, `competition` or `paid` |
| METACULUS_OPENROUTER_KEY | Secret | Dedicated organizer-issued key for competition mode |
| OPENROUTER_API_KEY | Secret | Dedicated separately funded key for paid mode |
| PER_QUESTION_USD / MONTHLY_USD | Variables | Both default to zero. Example opt-in credit limits are `0.02` and `2.00`. Limits use nominal inference dollars including donated credit. |
| ASKNEWS_API_KEY / ASKNEWS_COMPETITION_ACCESS | Secret / variable | Set the seasonal donated news key and `true` for latest-news retrieval. Use an account with paid overages disabled. Missing or unavailable search is disclosed in the explanation. |
| FORECAST_PROFILE | Variable | `baseline` (one GPT-4.1 mini estimate) or `experimental` (adds GPT-4.1 causal scenarios) |
| FORECAST_CALIBRATION | Variable | Binary log-odds multiplier, default `1`. Freeze any other value before evaluation. |
| SEASON_ID / MINIBENCH_ID | Variables | Default `33121` / `minibench` |

ChatGPT Pro and Claude Max subscriptions do not supply the API credentials used by this bot. The inference path pins the model and provider, sets a price ceiling and maximum completion length, reserves the maximum estimated charge before each call, and records reported tokens and cost. Ambiguous charges remain reserved. Provider funding fees, taxes, and unrelated use of the same provider account lie outside the inference ledger. A dedicated provider-side spending cap provides an additional limit. [OpenRouter routing](https://openrouter.ai/docs/features/provider-routing)

### Initialize an installation

With the GitHub CLI signed into the repository and the checkout on the deployed default branch, create the state branch and secret **only when absent**. The state workflow creates a new file and refuses to replace an existing journal. Keep a secure backup of the key if your organization requires recovery outside Actions. Never rotate the key without migrating the journal.

```sh
gh api -X POST repos/{owner}/{repo}/git/refs -f ref=refs/heads/futureeval-state -f sha="$(git rev-parse HEAD)"
uv run --locked python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' | gh secret set STATE_KEY
gh workflow run run_bot_on_tournament.yaml -f mode=init-state
```

Run `gh run list --workflow run_bot_on_tournament.yaml` and confirm initialization succeeded. Only encrypted data reaches `futureeval-state`. Each write checks the prior file SHA. An unreadable or conflicting journal prevents further authorized calls. The distributed guard assumes a single bot account, journal, and publishing workflow. Keep other clients using this bot token stopped.

### Monitor and stop

Each run uploads `health.json` with counts, runtime, revision, monthly settled and outstanding costs, and tokens. It also counts closed or resolved questions opened in the previous 24 hours without a forecast, including gaps when no job ran. Retention is 30 days. Unexpected errors and uncovered discovered posts fail the job. Read the Actions status and health counters during prize play. Examine explanations and tune prompts using synthetic, testing-area or permitted resolved material. Never adjust an individual open or upcoming tournament forecast or inspect its output to change the bot. [Tournament rules](https://www.metaculus.com/notebooks/28751/futureeval-competition-rules/)

To stop now, disable the workflow and cancel any running or queued instance. The runner also checks the workflow's active state before publishing. A request already received by the platform can still complete.

```sh
gh workflow disable run_bot_on_tournament.yaml
gh variable set FUTUREEVAL_ENABLED --body false
gh run list --workflow run_bot_on_tournament.yaml --limit 10
gh run cancel RUN_ID
```

To resume, enable the workflow and then set the production variable to true. When a forecast acknowledgement is lost, the job consults the authenticated user's forecast history on subsequent runs. It quarantines an ambiguous write if the remote history remains empty. This can forfeit coverage after a crash between recording intent and sending the request. Confirmed rate-limit rejections can reuse the saved payload. A private marker permits recovery of the required comment even after the question closes. Completed jobs remove the pending forecast content from the latest journal snapshot.

## Repository guide

- [runner.py](runner.py) handles discovery, deadline checks, submission, reconciliation and health reports.
- [formats.py](formats.py) interprets the resolution criteria and builds and validates Metaculus payloads.
- [pipeline.py](pipeline.py) implements dated research, independent estimation and private explanations.
- [state.py](state.py) persists encrypted submission state and cost reservations.
- [Sample environment](.env.template) lists settings. The runner reads exported environment variables. Actions maps repository settings automatically.
- [Competition API](https://www.metaculus.com/api/) documents account-specific data restrictions. Only tournament metadata, available question text and the bot's own state are needed. No Metaculus data is used for training.
