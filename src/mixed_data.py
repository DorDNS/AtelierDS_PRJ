#!/usr/bin/env python3
import pandas as pd
import csv
from pathlib import Path

# ─────────── chemins ───────────
# Exécuté depuis src/ ou notebooks/, on remonte d’un niveau → racine du projet
PROJECT_ROOT   = Path.cwd().resolve().parent
DATA_DIR       = PROJECT_ROOT / "AtelierDS_PRJ" / "data" / "processed"

FINAL_CSV      = DATA_DIR / "final_dataset.csv"
OLD_FINAL_CSV  = DATA_DIR / "old_final_dataset.csv"
MIXED_CSV      = DATA_DIR / "mixed_final_dataset.csv"

print("→ final_dataset.csv   :", FINAL_CSV.exists(), FINAL_CSV)
print("→ old_final_dataset.csv :", OLD_FINAL_CSV.exists(), OLD_FINAL_CSV)

# ─────────── chargement ───────────
# on désactive keep_default_na pour conserver les champs vides en "" avant le replace
df_new = (
    pd.read_csv(
        FINAL_CSV,
        parse_dates=["date", "closed_at", "cve_pub"],
        keep_default_na=False
    )
    .replace({"n/a": pd.NA, "": pd.NA})
)

df_old = (
    pd.read_csv(
        OLD_FINAL_CSV,
        parse_dates=["date", "closed_at", "cve_pub"],
        keep_default_na=False
    )
    .replace({"n/a": pd.NA, "": pd.NA})
)

# ─────────── concaténation + dédup ───────────
df_mixed = pd.concat([df_new, df_old], ignore_index=True)
before = len(df_mixed)
df_mixed = df_mixed.drop_duplicates().reset_index(drop=True)
after = len(df_mixed)

print(f"→ {before:,} lignes avant, {after:,} lignes après dé-dup (ajout de {after - len(df_new):,})")

# ─────────── écriture CSV en quoting=ALL ───────────
df_mixed.to_csv(
    MIXED_CSV,
    index=False,
    encoding="utf-8",
    quotechar='"',
    quoting=csv.QUOTE_ALL
)

print(f"✅ mix écrit dans {MIXED_CSV.name} ({after:,} lignes).")
