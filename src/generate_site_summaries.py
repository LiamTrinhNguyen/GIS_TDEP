"""
Generate three lightweight files for the web dashboard:

1. data/input/site_date_summary.csv
2. data/input/site_variable_coverage.json   (52-week aligned)
3. data/input/site_timeseries.json         (actual values for plotting)
"""

import json
from pathlib import Path
import polars as pl

BASE = Path("data/input")

CASTNET_WIDE = BASE / "CASTNET" / "CASTNET_transformed_wide.csv"
NADP_WIDE    = BASE / "NTN" / "NADP_transformed_wide.csv"

OUT_SUMMARY    = BASE / "site_date_summary.csv"
OUT_COVERAGE   = BASE / "site_variable_coverage.json"
OUT_TIMESERIES = BASE / "site_timeseries.json"

VARIABLES = [
    "variable_SO2", "variable_SO4", "variable_HNO3", "variable_NO3",
    "variable_TNO3", "variable_NH4", "variable_CA", "variable_K",
    "variable_MG", "variable_Sodium"
]


def process_network(csv_path: Path, network_name: str):
    print(f"Processing {network_name} ...")

    df = pl.read_csv(csv_path)

    # Parse dates
    if df["DATEON"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEON").str.to_date(strict=False))
    if "DATEOFF" in df.columns and df["DATEOFF"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEOFF").str.to_date(strict=False))

    df = df.with_columns([
        pl.col("DATEON").dt.year().alias("year"),
        pl.col("DATEON").dt.week().alias("week")
    ])

    # ----- 1. Date range summary -----
    summary = (
        df.group_by("SITE_ID")
          .agg([
              pl.col("DATEON").min().alias("first_date"),
              pl.col("DATEOFF").max().alias("last_date")
          ])
          .with_columns(pl.lit(network_name).alias("network"))
          .select(["SITE_ID", "network", "first_date", "last_date"])
    )

    # ----- 2. Coverage (always 52 weeks) + 3. Time series -----
    coverage = {}
    timeseries = {}

    for site_id, site_df in df.group_by("SITE_ID"):
        site_id = site_id[0]
        site_df = site_df.sort("DATEON")

        # ---- coverage ----
        site_coverage = {}
        for var in VARIABLES:
            short = var.replace("variable_", "")
            if var not in site_df.columns:
                site_coverage[short] = {}
                continue

            year_dict = {}
            for year, year_df in site_df.group_by("year"):
                year = int(year[0])
                flags = ["0"] * 52
                for row in year_df.iter_rows(named=True):
                    week = row["week"]
                    if week is None:
                        continue
                    week_idx = min(int(week), 52) - 1
                    if row[var] is not None:
                        flags[week_idx] = "1"
                year_dict[str(year)] = "".join(flags)
            site_coverage[short] = year_dict
        coverage[site_id] = site_coverage

        # ---- timeseries (actual values) ----
        ts = {"DATEON": [d.isoformat() if d else None for d in site_df["DATEON"].to_list()]}
        for var in VARIABLES:
            short = var.replace("variable_", "")
            if var in site_df.columns:
                # Convert to plain Python floats / None
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


def main():
    all_summaries = []
    all_coverage = {}
    all_timeseries = {}

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

    if all_summaries:
        pl.concat(all_summaries).write_csv(OUT_SUMMARY)
        print(f"Created: {OUT_SUMMARY}")

    with open(OUT_COVERAGE, "w") as f:
        json.dump(all_coverage, f)
    print(f"Created: {OUT_COVERAGE}")

    with open(OUT_TIMESERIES, "w") as f:
        json.dump(all_timeseries, f)
    print(f"Created: {OUT_TIMESERIES}")

    print("Done.")


if __name__ == "__main__":
    main()