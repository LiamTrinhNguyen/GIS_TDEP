"""
Generate lightweight files for the web dashboard:
1. data/input/site_date_summary.csv
2. data/input/site_variable_coverage.json  (52-week aligned; includes PPT/SUBPPT for NTN)
3. data/input/site_timeseries.json        (values for plotting; includes PPT/SUBPPT)
4. data/input/site_completeness.json      (site-level chemistry + optional PPT/SUBPPT metrics)

Completeness (per metric):
  span = weeks from first present to last present (inclusive)
  completeness_pct = 100 * n_record / n_span  (0 if no records)

For NTN, also computes by_variable completeness for PPT and SUBPPT
from columns PPT / SUBPPT (or variable_PPT / variable_SUBPPT) in the wide file.
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

# Chemistry (both networks when present in wide file)
CHEMISTRY_VARS = [
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

# NTN precipitation — stand-alone parameters (mm)
PRECIP_VARS = ["PPT", "SUBPPT"]


def short_name(col: str) -> str:
    return col.replace("variable_", "")


def bracket_from_pct(pct: float) -> str:
    if pct < 25:
        return "0-25"
    if pct < 50:
        return "25-50"
    if pct < 75:
        return "50-75"
    return "75-100"


def resolve_columns(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    """Return existing column names matching candidates (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    found = []
    for cand in candidates:
        if cand in df.columns:
            found.append(cand)
        elif cand.lower() in lower:
            found.append(lower[cand.lower()])
        elif f"variable_{cand}".lower() in lower:
            found.append(lower[f"variable_{cand}".lower()])
        elif cand.replace("variable_", "").lower() in lower:
            found.append(lower[cand.replace("variable_", "").lower()])
    # de-dupe preserve order
    out, seen = [], set()
    for c in found:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def parse_dates(df: pl.DataFrame) -> pl.DataFrame:
    if "DATEON" not in df.columns:
        raise ValueError("DATEON column required")
    if df["DATEON"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEON").str.to_date(strict=False))
    if "DATEOFF" in df.columns and df["DATEOFF"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("DATEOFF").str.to_date(strict=False))
    return df.with_columns(
        [
            pl.col("DATEON").dt.year().alias("year"),
            pl.col("DATEON").dt.week().alias("week"),
        ]
    )


def completeness_from_year_map(year_map: dict) -> dict:
    """year_map: {year: '0101...'} length 52. Compute span-based completeness."""
    flat = []
    years = sorted(year_map.keys())
    for y in years:
        s = year_map.get(y) or ""
        for ch in s:
            flat.append("1" if ch == "1" else "0")
    first = last = -1
    for i, ch in enumerate(flat):
        if ch == "1":
            if first < 0:
                first = i
            last = i
    if first < 0:
        return {
            "completeness_pct": 0.0,
            "bracket": "0-25",
            "n_record": 0,
            "n_missing": 0,
            "n_span": 0,
        }
    n_record = 0
    n_missing = 0
    for i in range(first, last + 1):
        if flat[i] == "1":
            n_record += 1
        else:
            n_missing += 1
    n_span = n_record + n_missing
    pct = (100.0 * n_record / n_span) if n_span else 0.0
    return {
        "completeness_pct": round(pct, 2),
        "bracket": bracket_from_pct(pct),
        "n_record": n_record,
        "n_missing": n_missing,
        "n_span": n_span,
    }


def process_network(csv_path: Path, network_name: str):
    print(f"Processing {network_name} ...")
    df = pl.read_csv(csv_path, infer_schema_length=10000)
    df = parse_dates(df)

    chem_cols = resolve_columns(df, CHEMISTRY_VARS)
    precip_cols = (
        resolve_columns(df, PRECIP_VARS + [f"variable_{p}" for p in PRECIP_VARS])
        if network_name == "NTN"
        else []
    )
    # Map actual column -> short name
    col_to_short = {}
    for c in chem_cols:
        col_to_short[c] = short_name(c)
    for c in precip_cols:
        s = short_name(c).upper()
        if s in ("PPT", "SUBPPT"):
            col_to_short[c] = s
        else:
            col_to_short[c] = short_name(c)

    print(f"  chemistry columns: {list(col_to_short.values())}")
    if precip_cols:
        print(f"  precip columns: {[col_to_short[c] for c in precip_cols]}")

    # ----- 1. Date range summary -----
    summary = (
        df.group_by("SITE_ID")
        .agg(
            [
                pl.col("DATEON").min().alias("first_DATEON"),
                pl.col("DATEON").max().alias("last_DATEOFF"),
            ]
        )
        .with_columns(pl.lit(network_name).alias("network"))
    )

    # ----- 2–4. Per-site coverage / timeseries / completeness -----
    coverage: dict = {}
    timeseries: dict = {}
    completeness: dict = {}

    site_ids = df["SITE_ID"].unique().to_list()
    for site_id in site_ids:
        sid = str(site_id)
        site_df = df.filter(pl.col("SITE_ID") == site_id).sort("DATEON")

        # timeseries base
        dates = site_df["DATEON"].to_list()
        date_strs = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in dates]
        ts: dict = {"DATEON": date_strs}

        site_cov: dict = {}
        by_variable_comp: dict = {}

        for col, short in col_to_short.items():
            vals = []
            for v in site_df[col].to_list():
                if v is None:
                    vals.append(None)
                else:
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(None)
            ts[short] = vals

            # year -> 52-char binary
            year_map: dict[str, list] = {}
            years = site_df["year"].to_list()
            weeks = site_df["week"].to_list()
            for y, w, val in zip(years, weeks, vals):
                if y is None or w is None:
                    continue
                yk = str(int(y))
                if yk not in year_map:
                    year_map[yk] = ["0"] * 52
                wi = int(w) - 1
                if wi < 0:
                    wi = 0
                if wi > 51:
                    wi = 51
                if val is not None:
                    year_map[yk][wi] = "1"
            site_cov[short] = {y: "".join(bits) for y, bits in year_map.items()}
            by_variable_comp[short] = completeness_from_year_map(site_cov[short])

        timeseries[sid] = ts
        coverage[sid] = site_cov

        # Site-level chemistry completeness: union of chemistry present weeks
        chem_shorts = [col_to_short[c] for c in chem_cols]
        chem_year_union: dict[str, list] = {}
        for short in chem_shorts:
            for y, bits in site_cov.get(short, {}).items():
                if y not in chem_year_union:
                    chem_year_union[y] = ["0"] * 52
                for i, ch in enumerate(bits):
                    if ch == "1":
                        chem_year_union[y][i] = "1"
        chem_map = {y: "".join(b) for y, b in chem_year_union.items()}
        chem_comp = completeness_from_year_map(chem_map)

        entry = {
            "completeness_pct": chem_comp["completeness_pct"],
            "bracket": chem_comp["bracket"],
            "n_record": chem_comp["n_record"],
            "n_missing": chem_comp["n_missing"],
            "n_span": chem_comp["n_span"],
            "by_variable": {},
        }
        for short in ("PPT", "SUBPPT"):
            if short in by_variable_comp:
                entry["by_variable"][short] = by_variable_comp[short]
        # also expose chemistry overall under metric key CHEM for UI clarity
        entry["by_variable"]["CHEM"] = {
            "completeness_pct": chem_comp["completeness_pct"],
            "bracket": chem_comp["bracket"],
            "n_record": chem_comp["n_record"],
            "n_missing": chem_comp["n_missing"],
            "n_span": chem_comp["n_span"],
        }
        completeness[sid] = entry

    return summary, coverage, timeseries, completeness


def main():
    all_summaries = []
    all_coverage = {}
    all_timeseries = {}
    all_completeness = {}

    if CASTNET_WIDE.exists():
        summary, coverage, timeseries, completeness = process_network(
            CASTNET_WIDE, "CASTNET"
        )
        all_summaries.append(summary)
        all_coverage["CASTNET"] = coverage
        all_timeseries["CASTNET"] = timeseries
        all_completeness["CASTNET"] = completeness
    else:
        print(f"WARNING: {CASTNET_WIDE} not found")

    if NADP_WIDE.exists():
        summary, coverage, timeseries, completeness = process_network(
            NADP_WIDE, "NTN"
        )
        all_summaries.append(summary)
        all_coverage["NTN"] = coverage
        all_timeseries["NTN"] = timeseries
        all_completeness["NTN"] = completeness
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

    with open(OUT_COMPLETENESS, "w") as f:
        json.dump(all_completeness, f)
    print(f"Created: {OUT_COMPLETENESS}")

    print("Done.")


if __name__ == "__main__":
    main()
