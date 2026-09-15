import csv
import hashlib
import json
from pathlib import Path

from audit_data import (
    EXPECTED_FILES,
    audit_country_alignment,
    audit_file,
    audit_raw_directory,
    run_audit,
    write_report_set,
)


ACTUAL_HEADER = (
    "Start date;End date;grid load [MWh] Calculated resolutions;"
    "Grid load incl. hydro pumped storage [MWh] Calculated resolutions;"
    "Hydro pumped storage [MWh] Calculated resolutions;"
    "Residual load [MWh] Calculated resolutions\n"
)

FORECAST_HEADER = (
    "Start date;End date;grid load [MWh] Calculated resolutions;"
    "Residual load [MWh] Calculated resolutions\n"
)


def write_csv(path: Path, rows: list[str], bom: bool = False) -> None:
    text = ACTUAL_HEADER + "\n".join(rows) + "\n"
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(text, encoding=encoding)


def write_forecast_csv(path: Path, rows: list[str]) -> None:
    path.write_text(FORECAST_HEADER + "\n".join(rows) + "\n", encoding="utf-8")


def valid_rows() -> list[str]:
    return [
        "Oct 24, 2020 11:00 PM;Oct 25, 2020 12:00 AM;100.00;100.00;0.00;90.00",
        "Oct 25, 2020 12:00 AM;Oct 25, 2020 1:00 AM;101.00;101.00;0.00;91.00",
        "Oct 25, 2020 1:00 AM;Oct 25, 2020 2:00 AM;102.00;102.00;0.00;92.00",
        "Oct 25, 2020 2:00 AM;Oct 25, 2020 3:00 AM;103.00;103.00;0.00;93.00",
        "Oct 25, 2020 2:00 AM;Oct 25, 2020 3:00 AM;103.00;103.00;0.00;93.00",
        "Oct 25, 2020 3:00 AM;Oct 25, 2020 4:00 AM;104.00;104.00;0.00;94.00",
    ]


def checks_by_id(result):
    return {check["check_id"]: check for check in result["checks"]}


def test_audit_returns_provenance_and_structured_core_results(tmp_path):
    source_path = tmp_path / "smard_germany_actual.csv"
    write_csv(source_path, valid_rows())
    timestamp = "2026-08-18T15:00:00+00:00"

    result = audit_file(
        source_path,
        country="Germany",
        dataset_type="actual",
        timezone_name="Europe/Berlin",
        audit_timestamp=timestamp,
    )
    checks = checks_by_id(result)

    assert result["provenance"] == {
        "filename": "smard_germany_actual.csv",
        "file_size_bytes": source_path.stat().st_size,
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "country": "Germany",
        "dataset_type": "actual",
        "timezone": "Europe/Berlin",
        "audit_timestamp": timestamp,
    }
    assert checks["provenance.file"]["status"] == "PASS"
    assert {
        "check_id",
        "scope",
        "status",
        "summary",
        "evidence",
    } <= checks["schema.headers"].keys()
    assert all(
        check["status"] in {"PASS", "WARNING", "ERROR"}
        for check in result["checks"]
    )


def test_audit_passes_valid_format_schema_intervals_and_expected_dst(tmp_path):
    source_path = tmp_path / "smard_germany_actual.csv"
    write_csv(source_path, valid_rows())

    result = audit_file(
        source_path,
        country="Germany",
        dataset_type="actual",
        timezone_name="Europe/Berlin",
    )
    checks = checks_by_id(result)

    assert checks["format.delimiter"]["status"] == "PASS"
    assert checks["format.field_counts"]["status"] == "PASS"
    assert checks["format.bom"]["status"] == "PASS"
    assert checks["schema.headers"]["status"] == "PASS"
    assert checks["schema.mapping"]["status"] == "PASS"
    assert checks["schema.shape"]["evidence"] == {
        "row_count": 6,
        "column_count": 6,
    }
    assert checks["schema.required_target"]["status"] == "PASS"
    assert checks["interval.coverage"]["evidence"] == {
        "first_interval_start_utc": "2020-10-24T21:00:00+00:00",
        "first_interval_end_utc": "2020-10-24T22:00:00+00:00",
        "final_interval_start_utc": "2020-10-25T02:00:00+00:00",
        "final_interval_end_utc": "2020-10-25T03:00:00+00:00",
    }
    assert checks["interval.utc_keys"]["status"] == "PASS"
    assert checks["interval.duration"]["status"] == "PASS"
    assert checks["interval.continuity"]["status"] == "PASS"
    assert checks["interval.dst"]["status"] == "PASS"
    assert checks["completeness.values"]["status"] == "PASS"
    assert checks["duplicates.rows"]["status"] == "PASS"
    assert checks["duplicates.rows"]["evidence"]["expected_autumn_rows"] == 2
    assert checks["duplicates.rows"]["evidence"]["unexpected_duplicate_rows"] == 0


def test_audit_reports_bom_and_numeric_diagnostics(tmp_path):
    source_path = tmp_path / "smard_austria_actual.csv"
    write_csv(
        source_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;;not-number;NaN;90.00"
        ],
        bom=True,
    )

    result = audit_file(
        source_path,
        country="Austria",
        dataset_type="actual",
        timezone_name="Europe/Vienna",
    )
    checks = checks_by_id(result)

    assert checks["format.bom"]["status"] == "PASS"
    assert checks["format.bom"]["evidence"] == {"present": True}
    assert checks["format.numeric_parsing"]["status"] == "PASS"
    assert checks["format.numeric_parsing"]["evidence"][
        "grid load [MWh] Calculated resolutions"
    ] == {"missing_count": 1, "non_numeric_count": 0, "non_finite_count": 0}
    assert checks["completeness.values"]["status"] == "ERROR"
    assert checks["completeness.values"]["evidence"]["grid_load_mwh"] == {
        "missing_count": 1,
        "empty_count": 1,
        "non_numeric_count": 0,
        "non_finite_count": 0,
    }


def test_audit_errors_on_unexpected_duplicate_rows_and_utc_gap(tmp_path):
    source_path = tmp_path / "smard_germany_actual.csv"
    write_csv(
        source_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;100.00;0.00;90.00",
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;100.00;0.00;90.00",
            "Jan 1, 2020 2:00 AM;Jan 1, 2020 3:00 AM;102.00;102.00;0.00;92.00",
        ],
    )

    result = audit_file(
        source_path,
        country="Germany",
        dataset_type="actual",
        timezone_name="Europe/Berlin",
    )
    checks = checks_by_id(result)

    assert checks["interval.utc_keys"]["status"] == "ERROR"
    assert checks["interval.continuity"]["status"] == "ERROR"
    assert checks["duplicates.rows"]["status"] == "ERROR"
    assert checks["duplicates.rows"]["evidence"]["unexpected_duplicate_rows"] == 2


def test_manifest_uses_the_four_confirmed_raw_filenames(tmp_path):
    expected_names = {
        "smard_austria_actual.csv",
        "smard_austria_forecasted.csv",
        "smard_germany_actual.csv",
        "smard_germany_forecasted.csv",
    }
    assert set(EXPECTED_FILES) == expected_names

    smard_directory = tmp_path / "smard"
    smard_directory.mkdir()
    for filename, metadata in EXPECTED_FILES.items():
        source_path = smard_directory / filename
        write_csv(source_path, valid_rows())

    results = audit_raw_directory(
        tmp_path,
        audit_timestamp="2026-08-18T15:00:00+00:00",
    )

    assert {result["provenance"]["filename"] for result in results} == expected_names
    assert {result["provenance"]["country"] for result in results} == {
        "Germany",
        "Austria",
    }


def test_audit_reports_descriptive_summaries_and_largest_hourly_change(tmp_path):
    source_path = tmp_path / "smard_germany_actual.csv"
    write_csv(
        source_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;102.00;2.00;-5.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;110.00;113.00;3.00;0.00",
            "Jan 1, 2020 2:00 AM;Jan 1, 2020 3:00 AM;90.00;90.00;0.00;10.00",
            "Jan 1, 2020 3:00 AM;Jan 1, 2020 4:00 AM;100.00;104.00;4.00;5.00",
        ],
    )

    checks = checks_by_id(
        audit_file(
            source_path,
            country="Germany",
            dataset_type="actual",
            timezone_name="Europe/Berlin",
        )
    )
    summary = checks["numerical.summaries"]

    assert summary["status"] == "PASS"
    grid_summary = summary["evidence"]["grid_load_mwh"]
    assert grid_summary["valid_count"] == 4
    assert grid_summary["minimum"] == 90.0
    assert grid_summary["maximum"] == 110.0
    assert grid_summary["mean"] == 100.0
    assert set(grid_summary["percentiles"]) == {
        "1",
        "5",
        "25",
        "50",
        "75",
        "95",
        "99",
    }
    assert grid_summary["largest_abs_one_hour_change"] == 20.0
    assert grid_summary["largest_change_interval_start_utc"] == (
        "2020-01-01T01:00:00+00:00"
    )
    assert checks["pumped_storage.relationship"]["status"] == "PASS"
    assert checks["pumped_storage.relationship"]["evidence"][
        "convention"
    ] == "plus"


def test_actual_target_and_confirmed_pumping_rules_use_errors_without_mutation(tmp_path):
    source_path = tmp_path / "smard_germany_actual.csv"
    rows = [
        "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;99.00;-1.00;0.00",
        "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;0.00;0.00;0.00;-2.00",
        "Jan 1, 2020 2:00 AM;Jan 1, 2020 3:00 AM;-1.00;0.00;1.00;0.00",
        "Jan 1, 2020 3:00 AM;Jan 1, 2020 4:00 AM;not-number;101.00;1.00;0.00",
    ]
    write_csv(source_path, rows)
    original_bytes = source_path.read_bytes()

    checks = checks_by_id(
        audit_file(
            source_path,
            country="Germany",
            dataset_type="actual",
            timezone_name="Europe/Berlin",
        )
    )

    assert checks["primary_target.validity"]["status"] == "ERROR"
    assert checks["primary_target.validity"]["evidence"][
        "invalid_observation_count"
    ] == 3
    assert checks["pumped_storage.relationship"]["status"] == "ERROR"
    assert checks["pumped_storage.relationship"]["evidence"][
        "convention"
    ] == "plus"
    assert checks["pumped_storage.relationship"]["evidence"][
        "negative_pumped_storage_rows"
    ]
    assert source_path.read_bytes() == original_bytes


def test_forecast_invalid_and_nonpositive_target_is_warning_and_unusable(tmp_path):
    source_path = tmp_path / "smard_germany_forecasted.csv"
    write_forecast_csv(
        source_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;90.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;0.00;90.00",
            "Jan 1, 2020 2:00 AM;Jan 1, 2020 3:00 AM;-1.00;90.00",
            "Jan 1, 2020 3:00 AM;Jan 1, 2020 4:00 AM;not-number;90.00",
        ],
    )

    checks = checks_by_id(
        audit_file(
            source_path,
            country="Germany",
            dataset_type="forecast",
            timezone_name="Europe/Berlin",
        )
    )
    validity = checks["primary_target.validity"]

    assert validity["status"] == "WARNING"
    assert validity["evidence"]["invalid_observation_count"] == 3
    assert len(validity["evidence"]["unusable_benchmark_observations"]) == 3
    assert checks["completeness.values"]["status"] == "WARNING"


def test_pumped_storage_convention_warning_does_not_reinterpret_values(tmp_path):
    source_path = tmp_path / "smard_austria_actual.csv"
    rows = [
        "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;150.00;2.00;90.00",
        "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;110.00;160.00;3.00;90.00",
    ]
    write_csv(source_path, rows)
    original_bytes = source_path.read_bytes()

    checks = checks_by_id(
        audit_file(
            source_path,
            country="Austria",
            dataset_type="actual",
            timezone_name="Europe/Vienna",
        )
    )
    relationship = checks["pumped_storage.relationship"]

    assert relationship["status"] == "WARNING"
    assert relationship["evidence"]["convention"] is None
    assert relationship["evidence"]["candidate_results"]["plus"][
        "mismatch_count"
    ] == 2
    assert relationship["evidence"]["candidate_results"]["minus"][
        "mismatch_count"
    ] == 2
    assert source_path.read_bytes() == original_bytes


def test_country_alignment_warns_for_missing_or_invalid_forecast_benchmark(
    tmp_path,
):
    actual_path = tmp_path / "smard_germany_actual.csv"
    forecast_path = tmp_path / "smard_germany_forecasted.csv"
    write_csv(
        actual_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;100.00;0.00;90.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;101.00;101.00;0.00;91.00",
            "Jan 1, 2020 2:00 AM;Jan 1, 2020 3:00 AM;102.00;102.00;0.00;92.00",
        ],
    )
    write_forecast_csv(
        forecast_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;90.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;0.00;90.00",
        ],
    )

    alignment = audit_country_alignment(
        actual_path,
        forecast_path,
        country="Germany",
        timezone_name="Europe/Berlin",
    )

    assert alignment["check_id"] == "alignment.country"
    assert alignment["scope"] == "country:Germany"
    assert alignment["status"] == "WARNING"
    assert alignment["evidence"]["missing_forecast_intervals"] == [
        "2020-01-01T01:00:00+00:00"
    ]
    assert alignment["evidence"]["invalid_forecast_observations"][0][
        "interval_start_utc"
    ] == "2020-01-01T00:00:00+00:00"
    assert alignment["evidence"]["forecast_role"] == "external_benchmark_only"


def test_country_alignment_errors_for_forecast_interval_without_actual_target(
    tmp_path,
):
    actual_path = tmp_path / "smard_austria_actual.csv"
    forecast_path = tmp_path / "smard_austria_forecasted.csv"
    write_csv(
        actual_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;100.00;0.00;90.00",
        ],
    )
    write_forecast_csv(
        forecast_path,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;90.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;101.00;90.00",
        ],
    )

    alignment = audit_country_alignment(
        actual_path,
        forecast_path,
        country="Austria",
        timezone_name="Europe/Vienna",
    )

    assert alignment["scope"] == "country:Austria"
    assert alignment["status"] == "ERROR"
    assert alignment["evidence"]["missing_actual_intervals"] == [
        "2020-01-01T00:00:00+00:00"
    ]


def write_project_files(project_root: Path, warning: bool = False, error: bool = False):
    raw_directory = project_root / "data" / "raw" / "smard"
    report_directory = project_root / "results" / "tables"
    raw_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    actual_rows = [
        "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;102.00;2.00;90.00",
        "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;101.00;101.00;0.00;91.00",
    ]
    forecast_rows = [
        "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;90.00",
        "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;101.00;91.00",
    ]
    if warning:
        forecast_rows[1] = (
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;0.00;91.00"
        )
    if error:
        forecast_rows.append(
            "Jan 1, 2020 2:00 AM;Jan 1, 2020 3:00 AM;102.00;92.00"
        )

    for country in ("austria", "germany"):
        write_csv(raw_directory / f"smard_{country}_actual.csv", actual_rows)
        write_forecast_csv(
            raw_directory / f"smard_{country}_forecasted.csv", forecast_rows
        )


def test_report_set_writes_exactly_three_serialized_reports(tmp_path):
    output_directory = tmp_path / "results" / "tables"
    output_directory.mkdir(parents=True)

    write_report_set(
        output_directory,
        "# SMARD Data Audit\n",
        '{"overall_status": "PASS"}\n',
        "file,country\nexample.csv,Germany\n",
    )

    assert {path.name for path in output_directory.iterdir()} == {
        "smard_data_audit.md",
        "smard_data_audit.json",
        "smard_column_summary.csv",
    }
    assert json.loads(
        (output_directory / "smard_data_audit.json").read_text(encoding="utf-8")
    )["overall_status"] == "PASS"


def test_report_write_failure_preserves_previous_reports(tmp_path):
    output_directory = tmp_path / "results" / "tables"
    output_directory.mkdir(parents=True)
    previous = {
        "smard_data_audit.md": "previous markdown",
        "smard_data_audit.json": '{"previous": true}',
        "smard_column_summary.csv": "previous csv",
    }
    for filename, contents in previous.items():
        (output_directory / filename).write_text(contents, encoding="utf-8")

    try:
        write_report_set(output_directory, "new markdown", "not json", "new csv")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid JSON should prevent report replacement")

    assert {
        filename: (output_directory / filename).read_text(encoding="utf-8")
        for filename in previous
    } == previous
    assert {path.name for path in output_directory.iterdir()} == set(previous)


def test_run_audit_writes_complete_json_and_column_summary_for_pass(tmp_path, capsys):
    write_project_files(tmp_path)

    exit_code = run_audit(
        project_root=tmp_path,
        audit_timestamp="2026-08-18T15:00:00+00:00",
    )
    output = capsys.readouterr().out
    report_directory = tmp_path / "results" / "tables"
    report = json.loads(
        (report_directory / "smard_data_audit.json").read_text(encoding="utf-8")
    )
    column_rows = list(
        csv.DictReader(
            (report_directory / "smard_column_summary.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )

    assert exit_code == 0
    assert "SMARD audit: PASS" in output
    assert "Files: 4" in output
    assert report["report_schema_version"] == "1.0"
    assert len(report["datasets"]) == 4
    assert len(report["alignments"]) == 2
    assert all(
        {"check_id", "scope", "status", "summary", "evidence"}
        <= check.keys()
        for dataset in report["datasets"]
        for check in dataset["checks"]
    )
    assert all(
        {"check_id", "scope", "status", "summary", "evidence"}
        <= check.keys()
        for check in report["alignments"]
    )
    assert len(column_rows) == 12
    assert {
        "file",
        "country",
        "dataset_type",
        "original_column",
        "standardized_column",
        "role",
        "unit",
        "source_dtype",
        "parsed_dtype",
        "row_count",
        "valid_value_count",
        "missing_count",
        "non_numeric_count",
        "non_finite_count",
        "minimum",
        "p1",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "mean",
        "maximum",
        "status",
    } <= set(column_rows[0])
    assert (report_directory / "smard_data_audit.md").read_text(
        encoding="utf-8"
    ).startswith("# SMARD Data Audit")
    assert {path.name for path in report_directory.iterdir()} == {
        "smard_data_audit.md",
        "smard_data_audit.json",
        "smard_column_summary.csv",
    }


def test_run_audit_returns_zero_for_warning_and_one_for_error(tmp_path, capsys):
    write_project_files(tmp_path, warning=True)
    warning_exit_code = run_audit(
        project_root=tmp_path,
        audit_timestamp="2026-08-18T15:00:00+00:00",
    )
    warning_output = capsys.readouterr().out

    assert warning_exit_code == 0
    assert "SMARD audit: WARNING" in warning_output

    write_project_files(tmp_path, error=True)
    error_exit_code = run_audit(
        project_root=tmp_path,
        audit_timestamp="2026-08-18T15:00:00+00:00",
    )
    error_output = capsys.readouterr().out

    assert error_exit_code == 1
    assert "SMARD audit: ERROR" in error_output
    assert (tmp_path / "results" / "tables" / "smard_data_audit.json").exists()


def test_run_audit_returns_two_and_preserves_reports_on_execution_failure(tmp_path, capsys):
    report_directory = tmp_path / "results" / "tables"
    report_directory.mkdir(parents=True)
    previous = {
        "smard_data_audit.md": "previous markdown",
        "smard_data_audit.json": '{"previous": true}',
        "smard_column_summary.csv": "previous csv",
    }
    for filename, contents in previous.items():
        (report_directory / filename).write_text(contents, encoding="utf-8")
    raw_directory = tmp_path / "data" / "raw" / "smard"
    raw_directory.mkdir(parents=True)
    write_csv(raw_directory / "smard_germany_actual.csv", valid_rows())

    exit_code = run_audit(
        project_root=tmp_path,
        audit_timestamp="2026-08-18T15:00:00+00:00",
    )
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "SMARD audit failed" in output
    assert {
        filename: (report_directory / filename).read_text(encoding="utf-8")
        for filename in previous
    } == previous
    assert {path.name for path in report_directory.iterdir()} == set(previous)
