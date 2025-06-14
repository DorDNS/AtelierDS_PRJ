"""
Fusionne les entrées ANSSI avec les métadonnées MITRE & EPSS,
calcule days_open et renvoie un DataFrame final.
"""

import os
import json
from datetime import datetime
import pandas as pd
from typing import Any, Dict, List


# ───────────────────────── utilitaires ──────────────────────────
def safe_get(d: Any, keys: List[Any], default=None):
    """Accès sécurisé à une structure imbriquée."""
    for key in keys:
        try:
            d = d[key]
        except (KeyError, IndexError, TypeError):
            return default
    return d


def clean_row(row_dict: Dict) -> Dict:
    """Nettoie les valeurs None, listes, retours à la ligne."""
    clean = {}
    for k, v in row_dict.items():
        if v is None:
            clean[k] = "n/a"
        elif isinstance(v, list):
            clean[k] = " | ".join(str(i) for i in v) if v else "n/a"
        elif k == "description":
            clean[k] = " ".join(str(v).split())
        else:
            clean[k] = v
    return clean


# ───────────────────── fonction principale ──────────────────────
def build_dataframe(
    entries: List[Dict],
    mitre_dir: str = "data/raw/mitre",
    epss_dir: str = "data/raw/first",
) -> pd.DataFrame:
    rows = []

    for entry in entries:
        cve = entry.get("cve")
        if not cve:
            continue

        # chemins vers les JSON MITRE & EPSS
        mitre_path = os.path.join(mitre_dir, cve)
        epss_path  = os.path.join(epss_dir,  cve)

        # lien dynamique (fallback)
        lien_dyn = f"https://www.cert.ssi.gouv.fr/{'alerte' if entry.get('type')=='alerte' else 'avis'}/{entry.get('id_anssi')}/"

        # --- MITRE ---------------------------------------------------------
        if not os.path.exists(mitre_path):
            continue
        try:
            with open(mitre_path, encoding="utf-8") as f:
                mitre_data = json.load(f)
        except Exception:
            continue

        cna = safe_get(mitre_data, ["containers", "cna"], {})

        # description anglaise
        description = next(
            (d["value"] for d in cna.get("descriptions", [])
             if d.get("lang") == "en"),
            None,
        )

        # CVSS (on récupère la première métrique v3.x trouvée)
        cvss_score = base_severity = None
        for metric in cna.get("metrics", []):
            for key in ("cvssV3_1", "cvssV3_0", "cvssV3"):
                if key in metric:
                    cvss = metric[key]
                    cvss_score    = cvss.get("baseScore")
                    base_severity = cvss.get("baseSeverity")
                    break
            if cvss_score is not None:
                break

        # CWE
        cwe = cwe_description = None
        cwe_list = safe_get(cna, ["problemTypes", 0, "descriptions"], [])
        if cwe_list:
            cwe             = cwe_list[0].get("cweId") or cwe_list[0].get("description")
            cwe_description = cwe_list[0].get("description")

        # EPSS
        epss_score = None
        if os.path.exists(epss_path):
            try:
                with open(epss_path, encoding="utf-8") as f:
                    epss_json = json.load(f)
                raw = safe_get(epss_json, ["data", 0, "epss"])
                if raw is not None:
                    epss_score = float(raw)
            except Exception:
                pass

        # Produits affectés (si listés)
        affected = cna.get("affected", [])
        if not affected:
            affected = [{"vendor": "n/a", "product": "n/a", "versions": ["n/a"]}]

        for prod in affected:
            vendor   = prod.get("vendor", "n/a")
            produit  = prod.get("product", "n/a")
            raw_versions = prod.get("versions", [])
            versions = []
            for v in raw_versions:
                if isinstance(v, dict):
                    if v.get("status") in (None, "affected"):
                        versions.append(v.get("version", "n/a"))
                else:
                    versions.append(str(v))
            if not versions:
                versions = ["n/a"]

            row = {
                # ─── champs ANSSI ------------------------------------------------
                "id_anssi":   entry.get("id_anssi"),
                "type":       entry.get("type"),
                "titre":      entry.get("titre"),
                "date":       entry.get("date"),
                "closed_at":  entry.get("closed_at"),
                "lien":       entry.get("lien") or lien_dyn,
                # ─── champs CVE --------------------------------------------------
                "cve":            cve,
                "description":    description,
                "cvss_score":     cvss_score,
                "base_severity":  base_severity,
                "cwe":            cwe,
                "cwe_description": cwe_description,
                "epss_score":     epss_score,
                # ─── produit -----------------------------------------------------
                "vendor":   vendor,
                "produit":  produit,
                "versions": versions,
            }
            rows.append(clean_row(row))

    # ─── DataFrame final + calcul days_open ───────────────────────────────
    df = pd.DataFrame(rows).drop_duplicates()

    # dates -> datetime
    # Les chaînes ISO de l’ANSSI ont toujours AAAA-MM-JJ[THH:MM:SS...]
    clean_date      = df["date"].astype(str).str.slice(0, 10)
    clean_closed_at = df["closed_at"].astype(str).str.slice(0, 10)

    df["date"]      = pd.to_datetime(clean_date,      format="%Y-%m-%d", errors="coerce")
    df["closed_at"] = pd.to_datetime(clean_closed_at, format="%Y-%m-%d", errors="coerce")


    # closed_at manquant → aujourd’hui
    today = pd.Timestamp("now").normalize()
    df["closed_at"] = df["closed_at"].fillna(today)

    # durée d’ouverture en jours
    df["days_open"] = (df["closed_at"] - df["date"]).dt.days

    return df