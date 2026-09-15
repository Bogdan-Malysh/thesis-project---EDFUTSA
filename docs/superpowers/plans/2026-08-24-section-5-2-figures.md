# Section 5.2 Figure Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing daily-demand Figure 5.3 and rename the existing weekly heatmap to Figure 5.4 without changing any verified Section 5.2 CSV results or protected figures.

**Architecture:** Keep the existing daily and weekly calculations unchanged. Add one renderer in the existing Section 5.2 module that consumes the already computed `day_of_week_summary` frame directly for two country panels, and update only the weekly renderer’s output filenames. The repository’s existing weekly PNG/PDF will be moved to their Figure 5.4 names so their bytes remain unchanged; the new daily PNG/PDF will be generated from the verified summary CSV.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib Agg backend, pytest, Pillow.

---

### Task 1: Specify the new figure contract with failing tests

**Files:**
- Modify: `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py:27-43,147-173`

- [ ] **Step 1: Protect the verified daily and weekly CSVs and expect the new filenames**

Add `day_of_week_summary.csv` and `weekly_hourly_profile.csv` to `PROTECTED_OUTPUTS`. Replace the old weekly Figure 5.3 filenames with `figure_5_4_weekly_demand_patterns.png` and `.pdf`.

- [ ] **Step 2: Extend the isolated-output expectation to Figure 5.3**

Change `expected_names` to exactly:

```python
expected_names = {
    "day_of_week_summary.csv",
    "weekly_hourly_profile.csv",
    "figure_5_3_daily_demand_patterns.png",
    "figure_5_3_daily_demand_patterns.pdf",
    "figure_5_4_weekly_demand_patterns.png",
    "figure_5_4_weekly_demand_patterns.pdf",
}
```

Add the same image/PDF checks for `figure_5_3_daily_demand_patterns.*` as for the weekly figure. Keep the weekly checks, changing their paths to Figure 5.4.

- [ ] **Step 3: Run the focused test file and verify it fails for the missing implementation**

Run from the project root:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_2_daily_weekly.py" -q
```

Expected result: failure because the current module still writes Figure 5.3 as the weekly heatmap and does not create the daily figure. Do not alter the verified output files to make this test pass yet.

### Task 2: Implement the daily figure and Figure 5.4 path changes

**Files:**
- Modify: `code/03_exploratory_analysis/section_5_2_daily_weekly.py:347-424`
- Test: `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py`

- [ ] **Step 1: Add the minimal daily renderer before the existing heatmap renderer**

Add:

```python
def _render_daily_figure(summary: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.8),
        sharex=True,
        gridspec_kw={"hspace": 0.38},
    )
    figure.patch.set_facecolor("white")
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_summary = summary.loc[summary["country"].eq(country)].sort_values(
            "day_of_week"
        )
        x = country_summary["day_of_week"].to_numpy()
        means = country_summary["mean_daily_mean_grid_load_mwh"].to_numpy()
        first_quartile = country_summary["first_quartile_mwh"].to_numpy()
        third_quartile = country_summary["third_quartile_mwh"].to_numpy()
        axis.fill_between(x, first_quartile, third_quartile, color="#DCE6F1", alpha=0.9)
        axis.plot(
            x,
            means,
            color="#164A7B",
            marker="o",
            markersize=4,
            linewidth=1.8,
        )
        axis.set_title(
            f"{panel} {country}",
            loc="left",
            pad=8,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.set_ylabel("Mean daily grid load [MWh]", color=TEXT_COLOR)
        axis.set_xlim(-0.25, 6.25)
        axis.set_xticks(range(7), DAY_NAMES)
        axis.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["left"].set_color(GRID_COLOR)
        axis.spines["bottom"].set_color(GRID_COLOR)
    axes[-1].set_xlabel("Day of week", color=TEXT_COLOR, labelpad=8)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
```

The function must use the summary’s mean, first-quartile, and third-quartile columns directly; it must not read raw data or recalculate daily statistics.

- [ ] **Step 2: Update the orchestration output paths only**

After writing the existing two CSVs, call `_render_daily_figure` with:

```python
_render_daily_figure(
    day_summary,
    output_directory / "figure_5_3_daily_demand_patterns.png",
    output_directory / "figure_5_3_daily_demand_patterns.pdf",
)
```

Change the existing `_render_heatmaps` call to write:

```python
output_directory / "figure_5_4_weekly_demand_patterns.png",
output_directory / "figure_5_4_weekly_demand_patterns.pdf",
```

Do not change the calculation functions, CSV names, formatting, or input loading.

- [ ] **Step 3: Run the focused tests and verify the isolated outputs pass**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_2_daily_weekly.py" -q
```

Expected result: all focused tests pass, with six isolated output files and unchanged protected inputs/outputs.

### Task 3: Replace the repository weekly filenames and generate only Figure 5.3

**Files:**
- Rename: `outputs/chapter5/section_5_2/figure_5_3_weekly_demand_patterns.png` to `outputs/chapter5/section_5_2/figure_5_4_weekly_demand_patterns.png`
- Rename: `outputs/chapter5/section_5_2/figure_5_3_weekly_demand_patterns.pdf` to `outputs/chapter5/section_5_2/figure_5_4_weekly_demand_patterns.pdf`
- Create: `outputs/chapter5/section_5_2/figure_5_3_daily_demand_patterns.png`
- Create: `outputs/chapter5/section_5_2/figure_5_3_daily_demand_patterns.pdf`

- [ ] **Step 1: Rename the existing weekly files without regenerating them**

Verify both old files exist and the new names do not, then move each old file to its Figure 5.4 name. This preserves the existing weekly file bytes and avoids rewriting either verified CSV.

- [ ] **Step 2: Render only the new daily figure from the existing summary CSV**

Use the project interpreter to import `_render_daily_figure`, read `outputs/chapter5/section_5_2/day_of_week_summary.csv`, and write only the two Figure 5.3 files. Do not invoke `run_section_5_2_daily_weekly`, because that function also writes the existing CSV outputs.

- [ ] **Step 3: Confirm the output directory has no obsolete weekly filenames**

The Section 5.2 directory must contain the existing CSVs, Figure 5.2, the new Figure 5.3 daily PNG/PDF, and the renamed Figure 5.4 weekly PNG/PDF. The old `figure_5_3_weekly_demand_patterns.*` paths must be absent.

### Task 4: Verify data preservation, figure properties, and regression safety

**Files:**
- Verify: `code/03_exploratory_analysis/section_5_2_daily_weekly.py`
- Verify: `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py`
- Verify: Section 5.2 output files and protected Section 5.1/Figure 5.2 files

- [ ] **Step 1: Compare protected hashes before and after**

Confirm unchanged SHA-256 hashes for `hourly_demand_profile.csv`, `day_of_week_summary.csv`, `weekly_hourly_profile.csv`, and both Figure 5.2 files. Confirm the renamed Figure 5.4 files have the recorded pre-rename weekly hashes.

- [ ] **Step 2: Inspect both new PNGs and PDFs**

Use Pillow to verify both PNGs have at least 299 dpi and non-trivial dimensions; verify both PDFs have non-zero size. Confirm the daily figure has Germany above Austria, Monday-Sunday ordering, and separate panel y-scales by inspecting the renderer and generated image.

- [ ] **Step 3: Run the complete test suite and dependency check**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest -q
& ".venv\Scripts\python.exe" -m pip check
```

Expected result: the full suite passes and `pip check` reports no broken requirements. No Git commit is created because this project is not a Git repository and project instructions prohibit initializing one.

## Self-Review

- The daily figure uses only the verified summary statistics and does not alter preprocessing or calculation code.
- Germany and Austria remain separate panels, with Germany first and independent y-axes because `sharey` is not enabled.
- Monday-Sunday ordering comes from the existing `DAY_CODES`/`DAY_NAMES` contract.
- Existing weekly bytes are preserved by rename, while future executions write the weekly heatmap under Figure 5.4.
- Tests cover the new filenames, both figure artifacts, protected CSVs, protected Figure 5.2, and isolated execution.
- No raw files, dependencies, or unrelated outputs are changed.
