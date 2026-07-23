"""
Generate:
1. data/input/site_date_summary.csv
2. data/input/site_variable_coverage.json   (now grouped by year)
"""

import json
from pathlib import Path
import polars as pl

BASE = Path("data/input")

CASTNET_WIDE = BASE / "CASTNET" / "CASTNET_transformed_wide.csv"
NADP_WIDE    = BASE / "NTN" / "NADP_transformed_wide.csv"

OUT_SUMMARY  = BASE / "site_date_summary.csv"
OUT_COVERAGE = BASE / "site_variable_coverage.json"

VARIABLES = [
    "variable_SO2", "variable_SO4", "variable_HNO3", "variable_NO3",
    "variable_TNO3", "variable_NH4", "variable_CA", "variable_K",
    "variable_MG", "variable_Sodium"
]


def process_network(csv_path: Path, network_name: str):
    print(f"Processing {network_name} ...")

    df = pl.read_csv(csv_path)

    # Ensure DATEON is Date type
    if df["DATEON"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEON").str.to_date(strict=False))
    if "DATEOFF" in df.columns and df["DATEOFF"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEOFF").str.to_date(strict=False))

    # Add year column
    df = df.with_columns(pl.col("DATEON").dt.year().alias("year"))

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

    # ----- 2. Coverage grouped by year -----
    coverage = {}

    for site_id, site_df in df.group_by("SITE_ID"):
        site_id = site_id[0]
        site_df = site_df.sort("DATEON")

        site_coverage = {}

        for var in VARIABLES:
            short_name = var.replace("variable_", "")
            if var not in site_df.columns:
                site_coverage[short_name] = {}
                continue

            year_dict = {}
            for year, year_df in site_df.group_by("year"):
                year = year[0]
                year_df = year_df.sort("DATEON")

                flags = (
                    year_df.select(
                        pl.when(pl.col(var).is_not_null())
                          .then(1)
                          .otherwise(0)
                          .alias("flag")
                    )
                    .to_series()
                    .to_list()
                )
                year_dict[str(year)] = "".join(map(str, flags))

            site_coverage[short_name] = year_dict

        coverage[site_id] = site_coverage

    return summary, coverage


def main():
    all_summaries = []
    all_coverage = {}

    if CASTNET_WIDE.exists():
        summary, coverage = process_network(CASTNET_WIDE, "CASTNET")
        all_summaries.append(summary)
        all_coverage["CASTNET"] = coverage
    else:
        print(f"WARNING: {CASTNET_WIDE} not found")

    if NADP_WIDE.exists():
        summary, coverage = process_network(NADP_WIDE, "NTN")
        all_summaries.append(summary)
        all_coverage["NTN"] = coverage
    else:
        print(f"WARNING: {NADP_WIDE} not found")

    if all_summaries:
        pl.concat(all_summaries).write_csv(OUT_SUMMARY)
        print(f"Created: {OUT_SUMMARY}")

    with open(OUT_COVERAGE, "w") as f:
        json.dump(all_coverage, f)
    print(f"Created: {OUT_COVERAGE}")

    print("Done.")


if __name__ == "__main__":
    main()