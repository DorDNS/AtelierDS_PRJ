#!/usr/bin/env python3
import sys
import os
import json

# 1️⃣ Ajouter la racine du projet au chemin d’import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.extract_cves import extract_all_entries
from src.consolidate import build_dataframe

def main():
    print("🚀 Lancement du pipeline ANSSI simplifié…")

    # Étape 1 : extraction
    all_entries = extract_all_entries(
        avis_dir=os.path.join(PROJECT_ROOT, "data/raw/Avis"),
        alertes_dir=os.path.join(PROJECT_ROOT, "data/raw/alertes")
    )
    print(f"📥 {len(all_entries)} bulletins + CVE extraits")

    # Sauvegarde JSON intermédiaire
    out_json_dir = os.path.join(PROJECT_ROOT, "data/processed")
    os.makedirs(out_json_dir, exist_ok=True)
    entries_path = os.path.join(out_json_dir, "entries.json")
    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"✅ Entrées brutes sauvegardées → {entries_path}")

    # Étape 2 : consolidation MITRE + EPSS
    df = build_dataframe(
        entries=all_entries,
        mitre_dir=os.path.join(PROJECT_ROOT, "data/raw/mitre"),
        epss_dir=os.path.join(PROJECT_ROOT, "data/raw/first")
    )
    print(f"🧩 {len(df)} lignes consolidées")

    # Sauvegarde finale en CSV
    out_csv = os.path.join(out_json_dir, "final_dataset.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Dataset final sauvegardé → {out_csv}")

if __name__ == "__main__":
    main()