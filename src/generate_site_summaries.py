"""
Generate lightweight files for the web dashboard.
Calendar year + calendar week from Jan 1.
Year-boundary rule:
  week 53 of year Y  OR  week 1 of year Y+1  →  set both bits to 1
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(-1)

BASE = Path("data/input")
CASTNET_WIDE = BASE / "CASTNET" / "CASTNET_transformed_wide.csv"
NADP_WIDE = BASE / "NTN" / "NADP_transformed_wide.csv"

OUT_SUMMARY = BASE / "site_date_summary.csv"
OUT_COVERAGE = BASE / "site_variable_coverage.json"
OUT_TIMESERIES = BASE / "site_timeseries.json"
OUT_COMPLETENESS = BASE / "site_completeness.json"

JSON_DUMP_KW = dict(indent=2, sort_keys=True, ensure_ascii=False)

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
            ((pl.col("DATEON").dt.ordinal_day() - 1) // 7 + 1).alias("week"),
        ]
    )


def completeness_from_year_map(year_map: dict) -> dict:
    flat = []
    for y in sorted(year_map.keys()):
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
    n_record = n_missing = 0
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


def _ensure_year(year_map: dict, year: int) -> list:
    yk = str(int(year))
    if yk not in year_map:
        year_map[yk] = ["0"] * 53
    return year_map[yk]


def _set_bit(year_map: dict, year: int, week: int) -> None:
    bits = _ensure_year(year_map, year)
    wi = int(week) - 1
    if wi < 0:
        wi = 0
    if wi > 52:
        wi = 52
    bits[wi] = "1"


def mark_week(year_map: dict, year, week, present: bool) -> None:
    """
    Mark the sample week present.
    Boundary rule: week 53 of Y and week 1 of Y+1 are the same
    year-turn sample → fill 1 in both slots.
    """
    if year is None or week is None or not present:
        return
    y = int(year)
    w = int(week)
    if w < 1:
        w = 1
    if w > 53:
        w = 53

    _set_bit(year_map, y, w)

    if w == 53:
        _set_bit(year_map, y + 1, 1)
    elif w == 1:
        _set_bit(year_map, y - 1, 53)


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

    col_to_short = {}
    for c in chem_cols:
        col_to_short[c] = short_name(c)
    for c in precip_cols:
        s = short_name(c).upper()
        col_to_short[c] = s if s in ("PPT", "SUBPPT") else short_name(c)

    print(f"  chemistry columns: {list(col_to_short.values())}")
    if precip_cols:
        print(f"  precip columns: {[col_to_short[c] for c in precip_cols]}")

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

    coverage: dict = {}
    timeseries: dict = {}
    completeness: dict = {}

    for site_id in df["SITE_ID"].unique().to_list():
        sid = str(site_id)
        site_df = df.filter(pl.col("SITE_ID") == site_id).sort("DATEON")

        dates = site_df["DATEON"].to_list()
        date_strs = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in dates]
        ts: dict = {"DATEON": date_strs}

        site_cov: dict = {}
        by_variable_comp: dict = {}
        years = site_df["year"].to_list()
        weeks = site_df["week"].to_list()

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

            year_map: dict[str, list] = {}
            for y, w, val in zip(years, weeks, vals):
                mark_week(year_map, y, w, val is not None)
            site_cov[short] = {y: "".join(bits) for y, bits in year_map.items()}
            by_variable_comp[short] = completeness_from_year_map(site_cov[short])

        timeseries[sid] = ts
        coverage[sid] = site_cov

        chem_shorts = [col_to_short[c] for c in chem_cols]
        chem_year_union: dict[str, list] = {}
        for short in chem_shorts:
            for y, bits in site_cov.get(short, {}).items():
                if y not in chem_year_union:
                    chem_year_union[y] = ["0"] * 53
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
            "by_variable": dict(by_variable_comp),
        }
        entry["by_variable"]["CHEM"] = {
            "completeness_pct": chem_comp["completeness_pct"],
            "bracket": chem_comp["bracket"],
            "n_record": chem_comp["n_record"],
            "n_missing": chem_comp["n_missing"],
            "n_span": chem_comp["n_span"],
        }
        completeness[sid] = entry

    print(summary.sort("SITE_ID"))
    print(f"  sites: {len(coverage)}")
    if coverage:
        example = next(iter(coverage))
        print(f"  example site {example}: variables = {list(coverage[example].keys())}")

    return summary, coverage, timeseries, completeness


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **JSON_DUMP_KW)
    print(f"Created: {path}")


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

    write_json(OUT_COVERAGE, all_coverage)
    write_json(OUT_TIMESERIES, all_timeseries)
    write_json(OUT_COMPLETENESS, all_completeness)
    print("Done.")


if __name__ == "__main__":
    main()