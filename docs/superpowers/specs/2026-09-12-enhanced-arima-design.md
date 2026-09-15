# Enhanced ARIMA Robustness Extension Design

**Status:** Approved in chat; pending written-spec review

**Authoritative methodology:** `docs/chapter6.pdf`

## Goal

Implement an isolated enhanced-ARIMA layer for Germany and Austria. The layer
will test a fixed ordinary ARIMA candidate set and a separate
weekly-differenced ARIMA branch using 2024 screening and validation only. It
will select and freeze one enhanced specification per country, then evaluate
only those frozen specifications on 2025.

This is a post-hoc robustness extension motivated in part by diagnostics
observed after the original 2025 evaluation. The original frozen ARIMA remains
the primary, pre-specified thesis result. Enhanced ARIMA does not replace the
original model in the primary model ranking. Although 2025 observations and
metrics are not used numerically for enhanced candidate selection, this
extension's 2025 period is not an untouched confirmatory holdout; its results
are descriptive post-hoc robustness results.

The existing authoritative ARIMA implementation, its 2024 and 2025 outputs,
central tables, registry, and comparator outputs remain unchanged.

## Constraints and Non-goals

- Do not initialize Git, add a remote, upload data, or add dependencies.
- Treat `data/raw/` as immutable.
- Use ordinary Python files and the project interpreter.
- Use classical ARIMA methods only; no machine learning, Prophet,
  `pmdarima`, temperature inputs, calendar regressors, or official forecasts as
  model inputs.
- Do not modify the behavior of the existing ARIMA pipeline.
- Do not write enhanced results to `results/arima/` or update existing central
  model tables with enhanced results.
- Do not use 2025 observations or metrics numerically for candidate search,
  screening, worker selection, adequacy decisions, or freezing. 2025 may be
  accessed only after the enhanced 2024 freeze for the descriptive post-hoc
  robustness evaluation.
- The original frozen ARIMA remains the primary thesis result and enhanced
  ARIMA must not replace it in the primary model ranking.
- Germany and Austria are always selected and reported separately.

## Existing Components to Reuse

The enhanced layer will call, without changing their behavior:

- `code/04_models/common/forecasting_framework.py` for prepared data, country
  origins, origin-bounded information sets, UTC ordering, local-date handling,
  DST target extraction, and target-day metrics.
- `code/04_models/arima/arima_models.py` for `fit_arima`, its existing
  convergence/warning/root/finite-value checks, residual diagnostics, and
  `forecast_values`.

Ordinary enhanced candidates use `fit_arima` directly. The existing
`fit_arima` finite-input guard means it cannot be used unchanged for the
weekly branch once mapping-invalid observations are represented at their true
hourly positions. The weekly branch therefore has a gated missing-aware
adapter around the underlying statsmodels ARIMA path, using the same order,
trend, optimizer, warning, convergence, root, finite-value, and residual
diagnostic conventions wherever the missing-aware result supports them. The
adapter must pass the missing-time regression tests before any weekly
production run; otherwise the weekly branch stops and reports the blocker.

The existing ordinary ARIMA files remain authoritative and are read only where
needed for the audit or comparator import.

## Module Boundaries

Add focused modules beside the existing ARIMA modules:

- `enhanced_arima_models.py` owns branch specifications, ordinary fitting
  delegation, weekly local-occurrence transformation, reconstruction, and
  branch-specific metadata.
- `enhanced_arima_screening_2024.py` owns the original-pipeline audit, fixed
  candidate definitions, fixed calendar, and 2024 screening.
- `enhanced_arima_benchmark.py` owns the fixed saturated workload and worker
  count measurements.
- `enhanced_arima_validation_2024.py` owns resumable shortlisted validation,
  common-evaluation-set metrics, adequacy review, and the country-specific
  freeze.
- `enhanced_arima_test_2025.py` owns freeze verification, resumable 2025
  evaluation, common-evaluation-set metrics, and the separate final comparison.

Matching focused test modules will cover each boundary. No refactor of the
existing ARIMA modules is required.

## Branch Specifications

### Ordinary ARIMA

Evaluate exactly these orders, with `d=1` and the existing `trend_for_d(1)`
convention (`trend="n"`):

```text
(1,1,1)  (1,1,2)  (1,1,3)
(2,1,1)  (2,1,2)  (2,1,3)
(3,1,1)  (3,1,2)  (3,1,3)
(4,1,1)  (4,1,2)
```

The branch uses specification identifiers with an enhanced ordinary prefix,
for example `enhanced_ordinary_arima_p1_d1_q1`, so they cannot collide with
existing ARIMA identifiers.

### Weekly-differenced ARIMA

Evaluate exactly these ARIMA orders on the weekly-differenced series:

```text
(1,0,1)  (2,0,0)  (2,0,1)  (2,0,2)  (3,0,1)
```

The branch uses identifiers such as
`weekly_differenced_arima_p1_d0_q1`. The `d=0` model uses the existing
`trend_for_d(0)` convention (`trend="c"`). The branch and ordinary branch
remain separate in manifests, output directories, metrics, and selection
logic.

## Weekly Local-Time and DST Semantics

For every prepared row, the source key is:

```text
(local_date minus 7 calendar days, hour, local_occurrence)
```

The source is resolved from the prepared frame's existing local-date, hour,
and occurrence columns. UTC subtraction is not used as a fallback. A source
mapping is valid only when exactly one source row exists and its actual value
is available within the current origin-bounded information set.

For fitting, valid historical pairs form `delta_t = y_t - y_source`. The
transformed series is placed on the complete original UTC hourly index. A
mapping-invalid observation is represented by `NaN` at its original timestamp,
with an explicit mapping-issue record; it is never deleted or compressed out
of the sequence. The missing-aware fitting gate above must verify that the
statsmodels state-space dynamics retain the original elapsed-time spacing.

The existing `fit_arima` function cannot consume this representation because
it rejects non-finite arrays. A preimplementation probe with the installed
statsmodels 0.14.6 path accepted an indexed hourly series containing a missing
observation, retained all indexed state-space positions and hourly frequency,
and produced a different forecast from a `dropna()`-compressed control. This
supports the adapter approach but does not waive the required regression test
or the weekly-branch gate.

The regression test must assert the preserved state-space observation count,
the original hourly frequency and timestamp positions, and a forecast/control
difference attributable to the retained missing position, using a documented
strict tolerance such as `rtol=1e-10` and `atol=1e-12`. It must fail if
the weekly implementation uses `dropna()`, positional compression, or an
irregular sequence that changes the elapsed-time dynamics. If the
missing-aware adapter cannot pass this test while retaining the existing fit
and diagnostic conventions, implementation of the weekly branch stops and
reports the blocker; the ordinary branch is not silently substituted.

For a forecast path, the weekly ARIMA forecasts the deltas continuously across
the bridge and target path. Each load forecast is reconstructed as:

```text
y_hat_t = y_source_t + delta_hat_t
```

The source value for every forecasted timestamp must be an origin-available
actual whose `interval_end_utc <= forecast_origin_utc`. The implementation must
assert this condition and must not use recursively reconstructed forecasts as
weekly source values. No post-origin actual is used. If a legitimate horizon
case is discovered where this condition cannot hold, the weekly branch stops
for review rather than silently introducing recursive sources.

If any timestamp in an otherwise valid target day has a missing or ambiguous
source mapping, the whole target date is marked `weekly_lag_invalid`; no row
within that target day is silently dropped and no alternate occurrence or UTC
lag is substituted.

Weekly mapping-invalid target dates are derived programmatically for every
country and validation year by scanning every target date and every target
timestamp against the strict source-key rule. The implementation must not
hard-code the autumn date or assume that it is the only invalid date. Screening
and full-validation eligibility use the derived invalid-date set. The required
regression cases explicitly cover the spring-DST day, the week following
spring DST, the autumn-DST day, and the week following autumn DST for both
countries, recording whether each case is valid or invalid rather than
assuming a result. A read-only derivation on the current processed data found
`2024-04-07` and `2024-10-27` as invalid in both countries; these observations
are evidence for the tests, not hard-coded eligibility constants. Every
derived invalid date, reason, affected timestamp, source key, and coverage is
persisted.

The date is excluded from aggregate weekly screening and native validation
metrics only when its required source occurrence cannot be mapped uniquely.
Weekly screening candidates require at least 15 of the 16 calendar dates to be
valid. Full 2024 and 2025 outputs retain the derived invalid dates and cannot
report 100% coverage when any exist.

## Deterministic 2024 Screening Calendar

The same explicit calendar is used for both countries and is written before
screening:

```text
2024-01-01  New Year holiday / cold season
2024-01-08  Monday / cold season
2024-01-27  Saturday / cold season
2024-02-15  ordinary weekday
2024-03-31  spring-DST Sunday
2024-04-01  Easter Monday holiday
2024-05-09  Ascension holiday
2024-06-17  Monday / warm season
2024-07-25  warm-season weekday
2024-08-18  warm-season Sunday
2024-09-12  ordinary weekday
2024-10-03  Germany-specific public holiday
2024-10-26  Austria-specific public holiday / Saturday
2024-10-27  autumn-DST Sunday
2024-11-11  Monday
2024-12-25  Christmas holiday / cold season
```

All screening jobs use the existing country-specific origins: Germany at 18:00
local time on the preceding day and Austria at 08:00 local time on the
preceding day. Only target-day observations whose interval end is available at
the origin can affect a fit. Forecast paths remain continuous, while metrics
use target timestamps only.

## Screening and Selection

The first output phase audits the existing ordinary ARIMA candidate grid,
selected `(2,1,2)`, nearby orders actually tested, and required enhanced
orders that were not tested. The audit is read-only and reports source paths,
overlap, unexplored candidates, and authoritative-result references.

The screening phase fits or verified-reuses every ordinary candidate and fits
every weekly candidate at every shared calendar date for each country. It
records fit metadata, warnings, residual diagnostics, target forecasts, native
metrics, coverage, and weekly mapping issues. Eligibility uses the
programmatically derived weekly invalid-date set, not a fixed date assumption.

Within each branch and country, candidates are ranked by aggregate 2024 MAE
over valid screening target observations, followed by RMSE, MAPE, and fixed
specification order. Ordinary candidates require all 16 dates to be valid.
Weekly candidates require at least 15 valid dates. The output contains the top
3 ordinary and top 2 weekly candidates per country.

This branch-local screening rule does not make ordinary-versus-weekly claims;
that comparison occurs only after full 2024 validation using the common set.

## Reuse of Authoritative Candidate Artifacts

The audit maps enhanced ordinary orders to equivalent candidate-level
authoritative 2024 artifacts. Existing jobs, forecasts, diagnostics, and
metrics may be reused read-only only when equivalence is verified for country,
order, trend, fit settings, forecast origins, information cutoffs, target
timestamps, forecast values, diagnostics, and coverage. Reused rows are copied
into the enhanced branch output with their source path and a verified-reuse
status; their enhanced identifier remains separate from the original
identifier.

If any required candidate-level evidence is missing or equivalence cannot be
verified, the candidate is refit. Missing artifacts are never fabricated. No
weekly artifact is assumed equivalent to an ordinary artifact. Reuse changes
runtime only and cannot change metrics, eligibility, or selection rules.

## Common-Evaluation-Set Metrics

For each country, full-validation comparison creates the intersection of valid
target UTC timestamps across every shortlisted candidate. Native metrics are
also retained for each candidate using only its own valid target dates.

The following are persisted for every candidate:

- Native MAE, RMSE, MAPE, evaluated observations, coverage, and invalid dates.
- Common-set MAE, RMSE, MAPE, evaluated observations, common-set size, and
  common-set coverage.

Common-set MAE is the primary ordinary-versus-weekly comparison metric, then
common-set RMSE and MAPE. Native coverage is always reported and a weekly
result below 100% coverage is never described as equivalent to a complete
ordinary result.

The same rule applies to the final 2025 comparison. If a frozen weekly model
has a DST-invalid date, the final output includes native metrics and metrics on
the exact intersection of valid timestamps across the competing models.

The final comparison CSV must contain, separately for Germany and Austria:

- Original ARIMA `(2,1,2)`.
- The selected enhanced ARIMA specification, ordinary or
  weekly-differenced.
- Original selected SARIMA.
- Weekly seasonal naive.
- Holt-Winters.
- The official SMARD benchmark.

The enhanced row includes its native coverage, invalid dates, and common-set
metrics whenever the valid timestamp sets differ. The original ARIMA remains
the primary thesis comparator; enhanced rows are labeled descriptive post-hoc
robustness results.

## Benchmark and Full Validation

After screening, create one fixed 32-job benchmark workload containing only
real computationally valid jobs:

- 2 countries.
- 2 ordinary shortlisted specifications per country.
- 2 weekly shortlisted specifications per country.
- Preferred dates `2024-02-15`, `2024-03-31`, `2024-07-25`, and `2024-11-03`.
  The derived invalid-date set is checked before materialization; no invalid
  date may enter the workload. If a preferred date is derived invalid, the
  next date in a fixed chronological candidate list is selected
  deterministically and the final workload CSV records the choice.

The same workload is run for worker counts `1` through `8` when supported by
the runner. Unsupported requested counts receive explicit CSV records. Each
run records requested workers, active workers, elapsed seconds, jobs per
minute, failures, peak RSS, peak swap usage, maximum forecast difference from
the one-worker run, forecast equivalence, safe/unsafe status, and the reason.
Numerical-library thread limits are set to `1` for OMP, MKL, OpenBLAS, and
NumExpr. The fastest demonstrably safe count is persisted.

Safety requires no unexpected failures, finite outputs, duplicate-free keys,
correct target lengths, expected weekly invalidity only, and forecast
equivalence to the one-worker reference within a fixed documented tolerance.
A requested worker count is supported when it is a positive integer no greater
than the detected logical CPU count and can be started by the spawn-based
runner; resource exhaustion or output-integrity failures make that run unsafe
even when the count is otherwise supported.
The complete shortlisted 2024 validation is run once after this choice, using
the selected worker count and a deterministic manifest over all 2024 dates.

Full validation is origin-by-origin with expanding histories. A completed job
requires both its job row and forecast rows. All derived weekly invalid target
dates remain explicit terminal invalid records and reduce native coverage.

## Adequacy and Freeze

Before freezing, every valid job for a candidate must have eligible convergence,
finite parameters, finite information criteria, finite forecasts, and no
implementation or data failure. Every programmatically derived weekly mapping
invalidity is a separate terminal status and is not relabeled as a successful
forecast.

Residual ACF and Ljung-Box diagnostics must be present and reviewed for every
valid fit. Existing diagnostic flags are retained. A residual warning is not
silently ignored; it is recorded as a reviewed warning when applicable.
The `residual_diagnostics_reviewed` field means that all required diagnostic
rows for each valid fit were consumed by the adequacy report; residual flags
remain warnings and are counted rather than hidden.

One specification is frozen per country after full 2024 validation using:

1. Common-set MAE.
2. Common-set RMSE and MAPE.
3. Residual/convergence adequacy check.
4. Deterministic specification order as the final tie-breaker.

The freeze CSV includes branch, order, native/common metrics, valid and invalid
dates, coverage, convergence status, residual review status, and the exact
selection rule. 2025 evaluation is blocked unless this file contains one
complete freeze decision for each country.

## Checkpoint and Persistence Contract

All enhanced outputs are CSV files. Warning lists and structured diagnostics
may be serialized inside CSV fields, but no JSON-only result is required.

Every phase uses deterministic manifests, stable keys, atomic temporary-file
replacement, duplicate-free upserts, and resume behavior. Completed jobs and
terminal expected `weekly_lag_invalid` jobs are skipped; incomplete or
unexpectedly failed jobs are retried according to the phase policy. Expected
weekly mapping invalidity is never retried as if it were an implementation
failure. No final selection or freeze is written until its required validation
and coverage checks pass: ordinary screening must cover 16/16 dates, weekly
screening at least 15/16, and full validation may contain only the
programmatically derived weekly mapping-invalid dates.

The output root is:

```text
results/enhanced_arima/
  audit/
  specifications/
  screening_2024/ordinary/
  screening_2024/weekly_differenced/
  benchmark/
  validation_2024/ordinary/
  validation_2024/weekly_differenced/
  freeze_2024/
  test_2025/ordinary/
  test_2025/weekly_differenced/
  comparison/
```

Required top-level artifacts include:

- `audit/original_arima_audit.csv`
- `audit/enhanced_arima_run_manifest.csv`
- `specifications/enhanced_arima_candidate_definitions.csv`
- `specifications/screening_calendar_2024.csv`
- Branch-specific screening, validation, and 2025 job, forecast, diagnostic,
  summary, and weekly mapping-issue CSVs.
- `screening_2024/shortlist_2024.csv`
- `benchmark/workload.csv`, per-count results, `worker_benchmark_summary.csv`,
  and `worker_selection.csv`.
- `validation_2024/common_evaluation_timestamps.csv` and common/native summary
  CSVs.
- `freeze_2024/frozen_specifications_2024.csv` and
  `freeze_2024/adequacy_review_2024.csv`.
- `comparison/enhanced_arima_final_comparison_2025.csv`.

The reproducibility manifest at
`audit/enhanced_arima_run_manifest.csv` records Python, NumPy, pandas, SciPy,
and statsmodels versions; platform/OS; logical CPU count; physical and
available RAM when practical; OMP, MKL, OpenBLAS, and NumExpr thread settings;
selected worker count after benchmarking; UTC run timestamp; and relevant
enhanced-ARIMA source filenames. It is written initially and updated
atomically after worker selection.

## Verification

Before model execution, tests will cover:

- Exact branch candidate sets and identifier separation.
- Fixed 16-date calendar and 2024-only manifests.
- Country-specific origins and origin-bounded information sets.
- Strict weekly local occurrence mapping and no UTC fallback.
- Programmatic invalid-date derivation for the spring-DST day, the week
  following spring DST, the autumn-DST day, and the week following autumn DST.
- Spring 23-row handling, autumn repeated-occurrence invalidity, and no
  intra-day target-row dropping.
- Missing weekly transform positions preserve elapsed-time spacing and do not
  become an apparently contiguous compressed ARIMA sequence.
- Every weekly forecast source has `interval_end_utc <= forecast_origin_utc`,
  with no recursive weekly-source reconstruction.
- Future-actual perturbation invariance.
- Native/common-set metric correctness and coverage accounting.
- Checkpoint resume, retry behavior, atomic writes, and duplicate-free keys.
- Benchmark workload identity, resource metrics, and worker forecast
  equivalence.
- Convergence, finite-value, residual-review, and freeze gates.
- Refusal to evaluate 2025 before the 2024 freeze.
- The final comparison contains original ARIMA, enhanced ARIMA, original
  SARIMA, weekly seasonal naive, Holt-Winters, and SMARD for both countries.
- Verified read-only reuse of complete authoritative candidate artifacts and
  refitting when evidence is incomplete.
- Preservation of existing ARIMA files and central result tables.

The existing ARIMA and SARIMA test suites must continue to pass. Long-running
benchmark and validation commands will be launched outside the interactive
session, preferably in tmux, with CSV checkpoints available for resume.
