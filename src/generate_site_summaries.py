"""
Generate four lightweight files for the web dashboard:

1. data/input/site_date_summary.csv
2. data/input/site_variable_coverage.json   (52-week aligned binary flags)
3. data/input/site_timeseries.json         (actual values for plotting)
4. data/input/site_completeness.json       (precomputed % + brackets)

Completeness rule (same as weekly coverage UI):
  - Flatten weeks in year order
  - Span = first week with data → last week with data
  - completeness_pct = 100 * n_record / n_span
  - Site-level % = mean of variable-level % (variables with any data only)
  - Brackets: 0-25 | 25-50 | 50-75 | 75-100
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

BASE = Path("data/input")
CASTNET_WIDE = BASE / "CASTNET" / "CASTNET_transformed_wide.csv"
NADP_WIDE = BASE / "NTN" / "NADP_transformed_wide.csv"

OUT_SUMMARY = BASE / "site_date_summary.csv"
OUT_COVERAGE = BASE / "site_variable_coverage.json"
OUT_TIMESERIES = BASE / "site_timeseries.json"
OUT_COMPLETENESS = BASE / "site_completeness.json"

VARIABLES = [
    "variable_SO2",
    "variable_SO4",
    "variable_HNO3",
    "variable_NO3",
    "variable_TNO3",
    "variable_NH4",
    "variable_CA",
    "variable_K",
    "variable_MG",
    "variable_Sodium",
]


# ---------------------------------------------------------------------------
# Completeness helpers
# ---------------------------------------------------------------------------

def completeness_from_binary_years(year_map: dict) -> dict:
    """
    year_map: { "2018": "01011...", "2019": "..." }

    Returns n_record, n_missing, n_span, completeness_pct
    using only the span from first observed week → last observed week.
    """
    if not year_map:
        return {
            "n_record": 0,
            "n_missing": 0,
            "n_span": 0,
            "completeness_pct": 0.0,
        }

    years = sorted(year_map.keys(), key=lambda y: int(y))
    flat: list[str] = []
    for y in years:
        s = year_map.get(y) or ""
        flat.extend("1" if ch == "1" else "0" for ch in s)

    first = -1
    last = -1
    for i, ch in enumerate(flat):
        if ch == "1":
            if first < 0:
                first = i
            last = i

    if first < 0:
        return {
            "n_record": 0,
            "n_missing": 0,
            "n_span": 0,
            "completeness_pct": 0.0,
        }

    span = flat[first : last + 1]
    n_record = span.count("1")
    n_missing = span.count("0")
    n_span = n_record + n_missing
    pct = round(100.0 * n_record / n_span, 1) if n_span else 0.0

    return {
        "n_record": n_record,
        "n_missing": n_missing,
        "n_span": n_span,
        "completeness_pct": pct,
    }


def bracket_for(pct: float) -> str:
    """Map completeness % into fixed display brackets."""
    if pct < 25:
        return "0-25"
    if pct < 50:
        return "25-50"
    if pct < 75:
        return "50-75"
    return "75-100"


def build_site_completeness(coverage: dict) -> dict:
    """
    coverage structure (site_variable_coverage.json):
      coverage[network][site_id][variable][year] = binary string of length 52

    Output:
      out[network][site_id] = {
        completeness_pct, bracket,
        n_record, n_missing, n_span, n_variables,
        by_variable: { VAR: pct, ... }
      }
    """
    out: dict = {}

    for network, sites in (coverage or {}).items():
        out[network] = {}
        for site_id, vars_map in (sites or {}).items():
            by_var: dict = {}
            pcts: list[float] = []
            total_rec = 0
            total_miss = 0
            total_span = 0

            for var, year_map in (vars_map or {}).items():
                stats = completeness_from_binary_years(year_map or {})
                if stats["n_span"] == 0:
                    continue
                by_var[var] = stats["completeness_pct"]
                pcts.append(stats["completeness_pct"])
                total_rec += stats["n_record"]
                total_miss += stats["n_missing"]
                total_span += stats["n_span"]

            if pcts:
                site_pct = round(sum(pcts) / len(pcts), 1)
            else:
                site_pct = 0.0

            out[network][site_id] = {
                "completeness_pct": site_pct,
                "bracket": bracket_for(site_pct),
                "n_record": total_rec,
                "n_missing": total_miss,
                "n_span": total_span,
                "n_variables": len(pcts),
                "by_variable": by_var,
            }

    return out


# ---------------------------------------------------------------------------
# Network processing
# ---------------------------------------------------------------------------

def process_network(csv_path: Path, network_name: str):
    print(f"Processing {network_name} ...")
    df = pl.read_csv(csv_path)

    # Parse dates
    if df["DATEON"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEON").str.to_date(strict=False))
    if "DATEOFF" in df.columns and df["DATEOFF"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEOFF").str.to_date(strict=False))

    df = df.with_columns(
        [
            pl.col("DATEON").dt.year().alias("year"),
            pl.col("DATEON").dt.week().alias("week"),  # ISO week 1–53
        ]
    )

    # ----- 1. Date range summary -----
    summary = (
        df.group_by("SITE_ID")
        .agg(
            [
                pl.col("DATEON").min().alias("first_date"),
                pl.col("DATEOFF").max().alias("last_date")
                if "DATEOFF" in df.columns
                else pl.col("DATEON").max().alias("last_date"),
            ]
        )
        .with_columns(pl.lit(network_name).alias("network"))
        .select(["SITE_ID", "network", "first_date", "last_date"])
    )

    # ----- 2. Coverage (always 52 weeks) + 3. Time series -----
    coverage: dict = {}
    timeseries: dict = {}

    for site_id, site_df in df.group_by("SITE_ID"):
        site_id = site_id[0]
        site_df = site_df.sort("DATEON")

        # ---- coverage ----
        site_coverage: dict = {}
        for var in VARIABLES:
            short = var.replace("variable_", "")
            if var not in site_df.columns:
                site_coverage[short] = {}
                continue

            year_dict: dict = {}
            for year, year_df in site_df.group_by("year"):
                year = int(year[0])
                flags = ["0"] * 52
                for row in year_df.iter_rows(named=True):
                    week = row["week"]
                    if week is None:
                        continue
                    # Clamp ISO week into 0..51 (week 53 maps to index 51)
                    week_idx = min(max(int(week), 1), 52) - 1
                    if row[var] is not None:
                        flags[week_idx] = "1"
                year_dict[str(year)] = "".join(flags)
            site_coverage[short] = year_dict

        coverage[site_id] = site_coverage

        # ---- timeseries (actual values) ----
        ts = {
            "DATEON": [
                d.isoformat() if d is not None else None
                for d in site_df["DATEON"].to_list()
            ]
        }
        for var in VARIABLES:
            short = var.replace("variable_", "")
            if var in site_df.columns:
                vals = []
                for v in site_df[var].to_list():
                    if v is None:
                        vals.append(None)
                    else:
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(None)
                ts[short] = vals
            else:
                ts[short] = [None] * len(ts["DATEON"])
        timeseries[site_id] = ts

    return summary, coverage, timeseries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_summaries = []
    all_coverage: dict = {}
    all_timeseries: dict = {}

    if CASTNET_WIDE.exists():
        summary, coverage, timeseries = process_network(CASTNET_WIDE, "CASTNET")
        all_summaries.append(summary)
        all_coverage["CASTNET"] = coverage
        all_timeseries["CASTNET"] = timeseries
    else:
        print(f"WARNING: {CASTNET_WIDE} not found")

    if NADP_WIDE.exists():
        summary, coverage, timeseries = process_network(NADP_WIDE, "NTN")
        all_summaries.append(summary)
        all_coverage["NTN"] = coverage
        all_timeseries["NTN"] = timeseries
    else:
        print(f"WARNING: {NADP_WIDE} not found")

    # 1) date summary
    if all_summaries:
        # Normalize column names expected by index.html (FIRST_DATE / LAST_DATE / NETWORK)
        combined = pl.concat(all_summaries)
        # write with uppercase-friendly headers used by parseCSV in index
        combined = combined.rename(
            {
                "SITE_ID": "SITE_ID",
                "network": "NETWORK",
                "first_date": "FIRST_DATE",
                "last_date": "LAST_DATE",
            }
        )
        combined.write_csv(OUT_SUMMARY)
        print(f"Created: {OUT_SUMMARY}")

    # 2) coverage
    with open(OUT_COVERAGE, "w") as f:
        json.dump(all_coverage, f)
    print(f"Created: {OUT_COVERAGE}")

    # 3) timeseries
    with open(OUT_TIMESERIES, "w") as f:
        json.dump(all_timeseries, f)
    print(f"Created: {OUT_TIMESERIES}")

    # 4) completeness (from coverage we just built)
    all_completeness = build_site_completeness(all_coverage)
    with open(OUT_COMPLETENESS, "w") as f:
        json.dump(all_completeness, f, indent=2)
    print(f"Created: {OUT_COMPLETENESS}")

    # quick sanity print
    for net, sites in all_completeness.items():
        n = len(sites)
        if n == 0:
            print(f"  {net}: 0 sites")
            continue
        pcts = [v["completeness_pct"] for v in sites.values()]
        print(
            f"  {net}: {n} sites · "
            f"mean completeness {sum(pcts) / n:.1f}% · "
            f"min {min(pcts):.1f}% · max {max(pcts):.1f}%"
        )

    print("Done.")


if __name__ == "__main__":
    main()