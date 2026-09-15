# Electricity Demand Forecasting Thesis

Local Python project for a master's thesis on forecasting hourly national electricity demand in Germany and Austria with classical time-series methods.

The current phase covers project setup and a read-only integrity audit of four SMARD CSV files in `data/raw/smard/`. It does not perform preprocessing, exploratory analysis, or modelling.

## Python Environment

The project targets 64-bit CPython 3.12. Create the local environment from the project root:

```powershell
py -3.12 -m venv ".venv"
```

Verify the interpreter:

```powershell
& ".\.venv\Scripts\python.exe" -c "import platform, struct, sys; print(sys.version); print(platform.python_implementation()); print(struct.calcsize('P') * 8)"
```

Install the direct project dependencies:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

For the exact tested dependency closure, use `requirements.lock` instead:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.lock
```

Install `requirements-dev.txt` when running the test suite.

Use `.venv\Scripts\python.exe` for every project command.

## Tests

Run the data-audit tests from the project root:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code\01_data_audit
```

## Data Audit

Run the audit from the project root:

```powershell
& ".\.venv\Scripts\python.exe" code\01_data_audit\audit_data.py
```

The audit will generate exactly these files:

- `results/tables/smard_data_audit.md`: concise readable findings
- `results/tables/smard_data_audit.json`: complete structured results
- `results/tables/smard_column_summary.csv`: compact column-level summary

Exit codes:

- `0`: execution completed with only `PASS` and `WARNING` results
- `1`: execution completed with one or more audit `ERROR` results
- `2`: execution failed before trustworthy reports could be produced

Warnings do not cause a non-zero exit code. Audit figures are outside this phase.

## Data Safety

Files in `data/raw/` are immutable inputs. Project code must never edit, rename, move, overwrite, delete, impute, or clean them.
