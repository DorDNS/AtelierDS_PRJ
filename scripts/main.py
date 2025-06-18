#!/usr/bin/env python3
import sys, os, json, csv, time
from pathlib import Path

import pandas as pd

# ── Setup PYTHONPATH ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports extraction & consolidation ──────────────────────────────────
from src.extract_cves import extract_all_entries
from src.consolidate    import build_dataframe as consolidate_raw
from src.extract_data   import build_dataframe as consolidate_flux

def main():
    t0 = time.perf_counter()
    print("🚀 Pipeline ANSSI complet…\n")

    # ── Dirs & fichiers ────────────────────────────────────────────────
    RAW_AVIS = PROJECT_ROOT / "data" / "raw" / "Avis"
    RAW_ALE  = PROJECT_ROOT / "data" / "raw" / "alertes"
    OUT_DIR  = PROJECT_ROOT / "data" / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OLD_CSV   = OUT_DIR / "old_final_dataset.csv"
    FLUX_CSV  = OUT_DIR / "flux_final_dataset.csv"
    FINAL_CSV = OUT_DIR / "final_dataset.csv"

    # ── Étape 1 : raw extraction + MITRE/EPSS ───────────────────────────
    print("[1/3] Extraction ANSSI ‘raw’…")
    entries = extract_all_entries(str(RAW_AVIS), str(RAW_ALE))
    print(f"    • {len(entries):,} bulletins⇄CVE extraits")
    # json intermédiaire
    with open(OUT_DIR / "entries.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print("    • entries.json sauvegardé")

    print("    • Consolidation MITRE+EPSS (raw)…")
    df_old = consolidate_raw(
        entries=entries,
        mitre_dir=str(PROJECT_ROOT / "data" / "raw" / "mitre"),
        epss_dir =str(PROJECT_ROOT / "data" / "raw" / "first"),
    )
    if "versions" in df_old.columns:
        df_old = df_old.drop(columns=["versions"])
    df_old.to_csv(OLD_CSV, index=False, encoding="utf-8",
                  quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    print(f"    ✅ {OLD_CSV.name} ({len(df_old):,} lignes)\n")

    # ── Étape 2 : flux RSS/API extraction ────────────────────────────────
    print("[2/3] Extraction via flux RSS/API…")
    df_flux = consolidate_flux()
    if "versions" in df_flux.columns:
        df_flux = df_flux.drop(columns=["versions"])
    df_flux.to_csv(FLUX_CSV, index=False, encoding="utf-8",
                   quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    print(f"    ✅ {FLUX_CSV.name} ({len(df_flux):,} lignes)\n")

    # ── Étape 3 : concat + dé-dup → final_dataset.csv ────────────────────
    print("[3/3] Concaténation & dé-duplication…")
    df1 = pd.read_csv(OLD_CSV,  parse_dates=["date","closed_at","cve_pub"], keep_default_na=False)
    df2 = pd.read_csv(FLUX_CSV, parse_dates=["date","closed_at","cve_pub"], keep_default_na=False)

    total_before = len(df1) + len(df2)
    df_mix = pd.concat([df1, df2], ignore_index=True)
    df_mix = df_mix.drop_duplicates().reset_index(drop=True)
    total_after = len(df_mix)
    added = total_after - len(df_flux)

    df_mix.to_csv(FINAL_CSV, index=False, encoding="utf-8",
                  quotechar='"', quoting=csv.QUOTE_ALL)
    print(f"    • {total_before:,} lignes lues → {total_after:,} après dé-dup (+{added:,})")
    print(f"\n✅ Pipeline achevé en {time.perf_counter()-t0:.1f}s")
    print(f"   • {OLD_CSV.name}\n   • {FLUX_CSV.name}\n   • {FINAL_CSV.name}")

if __name__ == "__main__":
    main()
