#!/usr/bin/env python3
"""
Combine ACF + FFT tables into ONE file per network (Polars).

Join / uniqueness keys:
  SITE_ID, Start_Date, End_Date, Num_Years

Stacked groups:
  same3len, same5len, seg  → column analysis_type

Output columns kept:
  analysis_type, SITE_ID, Start_Date, End_Date, Num_Years
  + _Lag / _Period columns paired by species
    e.g. acf_CA_Lag, fft_CA_Period, acf_SO4_Lag, fft_SO4_Period, ...

Sort:
  SITE_ID A→Z, then Num_Years high → low

Output:
  data/output/ACF_FFT_combined/CASTNET_ACF_FFT_combined.csv
  data/output/ACF_FFT_combined/NTN_ACF_FFT_combined.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output" / "ACF_FFT_combined"

NETWORKS = {
    "CASTNET": INPUT_DIR / "ACF_FFT_CASTNET",
    "NTN": INPUT_DIR / "ACF_FFT_NTN",
}

GROUPS = ("same3len", "same5len", "seg")
KEY_COLS = ["SITE_ID", "Start_Date", "End_Date", "Num_Years"]

# Preferred species order (others append alphabetically)
SPECIES_ORDER = [
    "SO2", "SO4", "HNO3", "NO3", "TNO3", "NH4",
    "CA", "K", "MG", "Sodium", "NA",
]


def detect_kind(path: Path) -> str | None:
    name = path.name.upper()
    if "ACF" in name and "FFT" not in name:
        return "ACF"
    if "FFT" in name:
        return "FFT"
    return None


def detect_group(path: Path) -> str | None:
    name = path.name.lower()
    for g in GROUPS:
        if g in name:
            return g
    return None


def load_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=10000)
    df = df.rename({c: c.strip() for c in df.columns})

    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing key columns {missing}")

    exprs = [
        pl.col("SITE_ID").cast(pl.Utf8).str.strip_chars(),
        pl.col("Start_Date").cast(pl.Utf8).str.strip_chars(),
        pl.col("End_Date").cast(pl.Utf8).str.strip_chars(),
        pl.col("Num_Years").cast(pl.Float64, strict=False),
    ]
    if "Seg_ID" in df.columns:
        exprs.append(pl.col("Seg_ID").cast(pl.Utf8).str.strip_chars())

    return df.with_columns(exprs)


def find_pair(folder: Path, group: str) -> tuple[Path | None, Path | None]:
    acf_path = fft_path = None
    if not folder.is_dir():
        return None, None
    for p in folder.glob("*.csv"):
        if detect_group(p) != group:
            continue
        kind = detect_kind(p)
        if kind == "ACF":
            acf_path = p
        elif kind == "FFT":
            fft_path = p
    return acf_path, fft_path


def combine_pair(acf_path: Path, fft_path: Path, analysis_type: str) -> pl.DataFrame:
    """Outer-join ACF and FFT on SITE_ID, Start_Date, End_Date, Num_Years."""
    acf = load_csv(acf_path)
    fft = load_csv(fft_path)

    merged = acf.join(fft, on=KEY_COLS, how="full", coalesce=True, suffix="_fft")

    acf_cols = [c for c in merged.columns if c.startswith("acf_")]
    fft_cols = [c for c in merged.columns if c.startswith("fft_")]

    if acf_cols and fft_cols:
        merged = merged.with_columns(
            pl.when(pl.col(acf_cols[0]).is_not_null() & pl.col(fft_cols[0]).is_not_null())
            .then(pl.lit("ACF+FFT"))
            .when(pl.col(acf_cols[0]).is_not_null())
            .then(pl.lit("ACF_only"))
            .when(pl.col(fft_cols[0]).is_not_null())
            .then(pl.lit("FFT_only"))
            .otherwise(pl.lit("unknown"))
            .alias("merge_source")
        )
    else:
        merged = merged.with_columns(pl.lit("ACF+FFT").alias("merge_source"))

    merged = merged.with_columns(pl.lit(analysis_type).alias("analysis_type"))

    front = ["analysis_type"] + [c for c in KEY_COLS if c in merged.columns]
    if "Seg_ID" in merged.columns:
        front.append("Seg_ID")
    if "Seg_ID_fft" in merged.columns:
        front.append("Seg_ID_fft")

    acf_sorted = sorted(c for c in merged.columns if c.startswith("acf_"))
    fft_sorted = sorted(c for c in merged.columns if c.startswith("fft_"))
    other = [
        c
        for c in merged.columns
        if c not in front
        and c not in acf_sorted
        and c not in fft_sorted
        and c != "merge_source"
    ]
    ordered = front + acf_sorted + fft_sorted + other + ["merge_source"]
    return merged.select([c for c in ordered if c in merged.columns])


def unify_dtype(dtypes: list[pl.DataType]) -> pl.DataType:
    """Pick one dtype all frames can cast to."""
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )

    def is_numeric(d: pl.DataType) -> bool:
        return isinstance(d, numeric_types) or d in numeric_types

    def is_string(d: pl.DataType) -> bool:
        return d in (pl.Utf8, pl.String) or isinstance(d, (pl.Utf8, pl.String))

    has_num = any(is_numeric(d) for d in dtypes)
    has_str = any(is_string(d) for d in dtypes)

    if has_num and not has_str:
        return pl.Float64
    if has_str or (has_num and has_str):
        return pl.Utf8
    return dtypes[0]


def align_and_concat(pieces: list[pl.DataFrame]) -> pl.DataFrame:
    """Union columns and unify dtypes so vertical concat succeeds."""
    all_cols: list[str] = []
    seen: set[str] = set()
    for df in pieces:
        for c in df.columns:
            if c not in seen:
                seen.add(c)
                all_cols.append(c)

    col_dtypes: dict[str, list[pl.DataType]] = {c: [] for c in all_cols}
    for df in pieces:
        for c in df.columns:
            col_dtypes[c].append(df.schema[c])

    target_dtype = {c: unify_dtype(ds) for c, ds in col_dtypes.items() if ds}

    aligned: list[pl.DataFrame] = []
    for df in pieces:
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).alias(c) for c in missing])

        casts = []
        for c in all_cols:
            if c not in df.columns:
                continue
            tgt = target_dtype.get(c)
            if tgt is None:
                continue
            if df.schema.get(c) != tgt:
                casts.append(pl.col(c).cast(tgt, strict=False).alias(c))
        if casts:
            df = df.with_columns(casts)

        aligned.append(df.select(all_cols))

    return pl.concat(aligned, how="vertical")


def extract_species(col: str) -> str | None:
    """
    From acf_CA_Lag / fft_CA_Period / acf_Sodium_Lag → CA / Sodium
    """
    m = re.match(r"^(?:acf|fft)_(.+?)_(?:Lag|Period)$", col, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def select_lag_period_paired(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep identity cols + Lag/Period metrics, ordered as:
      analysis_type, SITE_ID, Start_Date, End_Date, Num_Years,
      acf_CA_Lag, fft_CA_Period,
      acf_SO4_Lag, fft_SO4_Period,
      ...
    """
    lag_period_cols = [
        c for c in df.columns
        if "_lag" in c.lower() or "_period" in c.lower()
    ]

    # Map species → {lag: col, period: col}
    by_species: dict[str, dict[str, str]] = {}
    for c in lag_period_cols:
        sp = extract_species(c)
        if sp is None:
            continue
        bucket = by_species.setdefault(sp, {})
        cl = c.lower()
        if cl.endswith("_lag") or "_lag" in cl:
            bucket["lag"] = c
        if cl.endswith("_period") or "_period" in cl:
            bucket["period"] = c

    # Species order: preferred list first, then remaining alpha
    preferred = [s for s in SPECIES_ORDER if s in by_species]
    rest = sorted(s for s in by_species if s not in preferred)
    species_list = preferred + rest

    keep: list[str] = []
    if "analysis_type" in df.columns:
        keep.append("analysis_type")
    for c in KEY_COLS:
        if c in df.columns:
            keep.append(c)

    for sp in species_list:
        bucket = by_species[sp]
        if "lag" in bucket and bucket["lag"] not in keep:
            keep.append(bucket["lag"])
        if "period" in bucket and bucket["period"] not in keep:
            keep.append(bucket["period"])

    # Any lag/period that did not match the pattern
    for c in lag_period_cols:
        if c not in keep:
            keep.append(c)

    return df.select(keep)


def process_network(network: str, folder: Path) -> Path | None:
    if not folder.is_dir():
        print(f"[SKIP] {network}: folder not found → {folder}")
        return None

    print(f"\n=== {network} ({folder}) ===")
    pieces: list[pl.DataFrame] = []

    for group in GROUPS:
        acf_path, fft_path = find_pair(folder, group)
        if acf_path is None or fft_path is None:
            print(f"  [{group}] missing ACF or FFT — skip")
            continue

        print(f"  [{group}] ACF ← {acf_path.name}")
        print(f"  [{group}] FFT ← {fft_path.name}")

        try:
            part = combine_pair(acf_path, fft_path, analysis_type=group)
        except Exception as e:
            print(f"  [{group}] ERROR: {e}")
            continue

        pieces.append(part)
        print(f"  [{group}] rows={part.height}")

    if not pieces:
        print(f"  No data for {network}")
        return None

    combined = align_and_concat(pieces)

    # Dedupe on uniqueness keys + analysis_type
    uniq_keys = ["analysis_type"] + KEY_COLS
    before = combined.height
    combined = combined.unique(subset=uniq_keys, keep="first")
    after = combined.height
    if before != after:
        print(f"  dedupe on {uniq_keys}: {before} → {after}")

    # Keep Lag/Period only, paired by species
    combined = select_lag_period_paired(combined)
    print(f"  columns kept: {combined.columns}")

    # Sort: SITE_ID A→Z, then Num_Years high → low
    combined = combined.sort(
        by=["SITE_ID", "Num_Years"],
        descending=[False, True],
    )

    out_path = OUTPUT_DIR / f"{network}_ACF_FFT_combined.csv"
    combined.write_csv(out_path)

    print(f"  → {out_path.name}  rows={combined.height}  cols={combined.width}")
    return out_path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for network, folder in NETWORKS.items():
        path = process_network(network, folder)
        if path is not None:
            written.append(path)

    print("\n=== DONE ===")
    if not written:
        print("No combined files written.")
        return 1

    print(f"Wrote {len(written)} file(s) (one per network):")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())