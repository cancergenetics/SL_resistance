"""Step 02 — Map mouse gene symbols to human orthologs (IC90 SUM only).

Primary: Ensembl BioMart ortholog table (downloaded once, cached).
Fallback 1: uppercase + HGNC approved symbol exact match.
Fallback 2: uppercase + HGNC alias_symbol / prev_symbol lookup.

Maps mouse_gene → human query_gene for the IC90 SUM label file:
  - Drops unmapped genes (human_gene is NaN)
  - Drops mouse_gene column; renames human_gene → query_gene
  - Writes human-label CSV: query_gene, screen_name, pos_rank, pos_lfc, pos_fdr, label

Outputs:
  data/mouse_to_human_map.csv          — full mapping table (all unique mouse genes)
  data/labels_KO_IC90_SUM_human.csv
"""

from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA,
    ENSEMBL_ORTHOLOGS_TSV,
    HGNC_TSV,
    KNOWN_HITS_HUMAN,
    KNOWN_HITS_MOUSE,
    LABELS_IC90_SUM_CSV,
    LABELS_IC90_SUM_HUMAN_CSV,
    MOUSE_HUMAN_MAP_CSV,
    SCREEN_IC90_SUM,
)

BIOMART_XML = (
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<!DOCTYPE Query>"
    "<Query virtualSchemaName='default' formatter='TSV' header='0' "
    "uniqueRows='1' count='' datasetConfigVersion='0.6'>"
    "<Dataset name='mmusculus_gene_ensembl' interface='default'>"
    "<Attribute name='external_gene_name'/>"
    "<Attribute name='hsapiens_homolog_associated_gene_name'/>"
    "</Dataset></Query>"
)
BIOMART_URL = ("https://mart.ensembl.org/biomart/martservice?query="
               + urllib.parse.quote(BIOMART_XML))


def download_ensembl_orthologs(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Ensembl BioMart orthologs → {out_path} …")
    with urllib.request.urlopen(BIOMART_URL, timeout=120) as r:
        data = r.read().decode("utf-8")
    out_path.write_text(data)
    print(f"  Downloaded: {len(data.splitlines()):,} lines")


def load_ensembl_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t", header=None, names=["mouse_gene", "human_gene"],
                     dtype=str)
    df = df[df["human_gene"].notna() & (df["human_gene"] != "")]
    df = df.drop_duplicates(subset="mouse_gene", keep="first")
    return dict(zip(df["mouse_gene"], df["human_gene"]))


def build_hgnc_lookups(hgnc_path: Path) -> tuple[set[str], dict[str, str], dict[str, str]]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", low_memory=False)
    approved = set(hgnc["symbol"].dropna())

    alias_map: dict[str, str] = {}
    prev_map:  dict[str, str] = {}
    for col, store in [("alias_symbol", alias_map), ("prev_symbol", prev_map)]:
        for sym, val in zip(hgnc["symbol"], hgnc[col]):
            if pd.isna(val):
                continue
            for alt in str(val).split("|"):
                alt = alt.strip()
                if alt and alt not in store:
                    store[alt] = sym
    return approved, alias_map, prev_map


def resolve_gene(
    mouse_gene: str,
    ensembl_map: dict[str, str],
    hgnc_approved: set[str],
    hgnc_alias: dict[str, str],
    hgnc_prev: dict[str, str],
) -> tuple[str | None, str]:
    if mouse_gene in ensembl_map:
        return ensembl_map[mouse_gene], "Ensembl"
    up = mouse_gene.upper()
    if up in hgnc_approved:
        return up, "HGNC_exact"
    # strict 3-tier: prev_symbol (former approved name) beats alias_symbol
    if up in hgnc_prev:
        return hgnc_prev[up], "HGNC_prev"
    if up in hgnc_alias:
        return hgnc_alias[up], "HGNC_alias"
    return None, "unmapped"


def apply_mapping(labels_csv: Path, human_csv: Path, mouse_to_human: dict[str, str],
                  screen_name_override: str) -> None:
    df = pd.read_csv(labels_csv)
    df["query_gene"] = df["mouse_gene"].map(mouse_to_human)
    df = df.dropna(subset=["query_gene"]).reset_index(drop=True)
    df["screen_name"] = screen_name_override
    out = df[["query_gene", "screen_name", "pos_rank", "pos_lfc", "pos_fdr", "label"]]
    out.to_csv(human_csv, index=False)
    counts = out["label"].value_counts()
    print(f"  {human_csv.name}: Resistance={counts.get('Resistance',0)}  "
          f"Non-Resistance={counts.get('Non-Resistance',0)}  "
          f"Total={len(out)}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    if not ENSEMBL_ORTHOLOGS_TSV.exists():
        download_ensembl_orthologs(ENSEMBL_ORTHOLOGS_TSV)
    else:
        print(f"  Ensembl orthologs cached: {ENSEMBL_ORTHOLOGS_TSV}")

    print("  Loading Ensembl map …")
    ensembl_map = load_ensembl_map(ENSEMBL_ORTHOLOGS_TSV)
    print(f"  Ensembl entries with human ortholog: {len(ensembl_map):,}")

    print("  Loading HGNC …")
    hgnc_approved, hgnc_alias, hgnc_prev = build_hgnc_lookups(HGNC_TSV)

    all_mouse_genes: set[str] = set(pd.read_csv(LABELS_IC90_SUM_CSV)["mouse_gene"])

    print(f"  Resolving {len(all_mouse_genes):,} unique mouse genes …")
    rows = []
    for g in sorted(all_mouse_genes):
        human, source = resolve_gene(g, ensembl_map, hgnc_approved, hgnc_alias, hgnc_prev)
        rows.append({"mouse_gene": g, "human_gene": human, "source": source})

    map_df = pd.DataFrame(rows)
    map_df.to_csv(MOUSE_HUMAN_MAP_CSV, index=False)

    src_counts = map_df["source"].value_counts()
    n_mapped   = int(map_df["human_gene"].notna().sum())
    n_total    = len(map_df)
    print(f"\nMapping summary ({n_total:,} genes):")
    for src, cnt in src_counts.items():
        print(f"  {src:15s}: {cnt:,}")
    print(f"  Mapped total : {n_mapped:,} ({100*n_mapped/n_total:.1f}%)")
    print(f"  Unmapped     : {n_total - n_mapped:,}")

    mouse_to_human = dict(zip(map_df["mouse_gene"], map_df["human_gene"]))

    print("\nWriting human-ortholog label file …")
    apply_mapping(LABELS_IC90_SUM_CSV, LABELS_IC90_SUM_HUMAN_CSV, mouse_to_human, SCREEN_IC90_SUM)

    print(f"\nKnown-hit ortholog check (IC90 SUM):")
    sum_map = pd.read_csv(LABELS_IC90_SUM_HUMAN_CSV).set_index("query_gene")
    src_lookup = map_df.set_index("mouse_gene")["source"]
    for m, h in zip(KNOWN_HITS_MOUSE, KNOWN_HITS_HUMAN):
        result = mouse_to_human.get(m)
        src    = src_lookup.get(m, "?")
        ok     = "✓" if result == h else "✗"
        in_res = "in_Resistance" if (result in sum_map.index and
                                     sum_map.at[result, "label"] == "Resistance") else "not_top100"
        print(f"  {m:8s} → {str(result):8s}  (expected {h})  {ok}  [{src}]  {in_res}")

    print(f"\n  Wrote {MOUSE_HUMAN_MAP_CSV}")


if __name__ == "__main__":
    main()
