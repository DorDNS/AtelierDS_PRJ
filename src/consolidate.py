"""
consolidate.py  –  v2
---------------------
Fusionne les entrées ANSSI (avis/alertes) avec les méta-données MITRE & EPSS,
enrichit le jeu : CVSS_sev, percentile EPSS, date publi CVE, délai de réaction
ANSSI, normalisation vendor ; calcule days_open puis renvoie un DataFrame.
"""
from __future__ import annotations

import json, os, csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ───────────────────────── utilitaires ──────────────────────────
def safe_get(d: Any, keys: List[Any], default=None):
    """Accès sécurisé à une structure imbriquée."""
    for k in keys:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return default
    return d


# --- petite table de correspondance Vendor ---------------------------------
_DEFAULT_VENDOR_MAP = {
    #   alias (minuscule)      forme canonique
    "microsoft corporation":   "Microsoft",
    "microsoft corp.":         "Microsoft",
    "microsoft":               "Microsoft",
    "google llc":              "Google",
    "google":                  "Google",
    "oracle corporation":      "Oracle",
    "oracle":                  "Oracle",
    "international business machines corporation": "IBM",
    "ibm":                     "IBM",
}
def load_vendor_aliases(csv_path: Path | str | None = None) -> Dict[str, str]:
    """
    Charge un CSV 2 colonnes alias,canonique si présent,
    sinon renvoie le mapping par défaut.
    """
    if csv_path and Path(csv_path).exists():
        alias_map: Dict[str, str] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for alias, canon in csv.reader(f):
                alias_map[alias.lower()] = canon
        return alias_map
    return _DEFAULT_VENDOR_MAP.copy()

_VENDOR_MAP = load_vendor_aliases("data/mappings/vendor_aliases.csv")

def normalize_vendor(vendor_raw: str | None) -> str | None:
    if vendor_raw is None:
        return None
    return _VENDOR_MAP.get(vendor_raw.lower(), vendor_raw)


def clean_row(row: Dict) -> Dict:
    """
    - Remplace '', 'n/a', []… par None
    - Aplati les listes en chaînes
    - Compacte les descriptions
    """
    out = {}
    for k, v in row.items():
        if v in ("n/a", "", [], {}):
            out[k] = None
        elif isinstance(v, list):
            out[k] = " | ".join(map(str, v)) if v else None
        elif k == "description":
            out[k] = " ".join(str(v).split())
        else:
            out[k] = v
    return out

# ───────────────────── fonction principale ──────────────────────
def build_dataframe(
    entries: List[Dict],
    mitre_dir: str = "data/raw/mitre",
    epss_dir: str = "data/raw/first",
) -> pd.DataFrame:
    rows: list[dict] = []

    for e in entries:
        cve = e.get("cve")
        if not cve:
            continue

        # chemins
        mitre_path = Path(mitre_dir) / cve
        epss_path  = Path(epss_dir)  / cve

        # fallback URL bulletin
        lien_dyn = f"https://www.cert.ssi.gouv.fr/{'alerte' if e.get('type')=='alerte' else 'avis'}/{e.get('id_anssi')}/"

        # ─── JSON MITRE (obligatoire) ──────────────────────────────────
        if not mitre_path.exists():
            continue
        try:
            mitre = json.load(open(mitre_path, encoding="utf-8"))
        except Exception:
            continue

        cna = safe_get(mitre, ["containers", "cna"], {})

        desc = next((d["value"] for d in cna.get("descriptions", []) if d.get("lang") == "en"), None)

        # ---- CVSS (v3 prioritaire) -----------------------------------
        cvss_score = base_severity = None
        for metric in cna.get("metrics", []):
            for key in ("cvssV3_1", "cvssV3_0", "cvssV3", "cvssV2"):
                if key in metric:
                    block         = metric[key]
                    cvss_score    = block.get("baseScore")
                    base_severity = block.get("baseSeverity")
                    break
            if cvss_score is not None:
                break

        # ---- CWE -----------------------------------------------------
        cwe = cwe_desc = None
        pt = safe_get(cna, ["problemTypes", 0, "descriptions"], [])
        if pt:
            cwe      = pt[0].get("cweId") or pt[0].get("description")
            cwe_desc = pt[0].get("description")

        # ---- Nombre de références CVE --------------------------------
        references = cna.get("references", [])
        n_cve_refs = len(references) if references else 0

        # ---- EPSS ----------------------------------------------------
        epss_score = epss_percentile = None
        if epss_path.exists():
            try:
                epss_json = json.load(open(epss_path, encoding="utf-8"))
                epss_score       = float(safe_get(epss_json, ["data", 0, "epss"]))
                epss_percentile  = float(safe_get(epss_json, ["data", 0, "percentile"]))
            except Exception:
                pass

        # ---- Date publication CVE & lag ------------------------------
        cve_pub = safe_get(mitre, ["cveMetadata", "datePublished"])
        if cve_pub:
            cve_pub = cve_pub[:10]  # AAAA-MM-JJ
            try:
                lag_anssi = (
                    pd.to_datetime(e["date"][:10]) -
                    pd.to_datetime(cve_pub)
                ).days
            except Exception:
                lag_anssi = None
        else:
            lag_anssi = None

        # ---- Produits / vendors --------------------------------------
        affected = cna.get("affected", []) or [{"vendor": "n/a", "product": "n/a", "versions": ["n/a"]}]
        for prod in affected:
            vendor_raw = prod.get("vendor", "n/a")
            vendor_std = normalize_vendor(vendor_raw)
            produit    = prod.get("product", "n/a")
            versions   = [
                (v.get("version") if isinstance(v, dict) else str(v))
                for v in prod.get("versions", [])
            ] or ["n/a"]

            rows.append(
                clean_row({
                    # ─── méta ANSSI ──────────────────────────────
                    "id_anssi":   e.get("id_anssi"),
                    "type":       e.get("type"),
                    "titre":      e.get("titre"),
                    "date":       e.get("date"),
                    "closed_at":  e.get("closed_at"),
                    "lien":       e.get("lien") or lien_dyn,
                    # ─── méta CVE / MITRE ───────────────────────
                    "cve":            cve,
                    "description":    desc,
                    "cvss_score":     cvss_score,
                    "cvss_sev":       base_severity,
                    "cwe":            cwe,
                    "cwe_description":cwe_desc,
                    "n_cve_refs":     n_cve_refs,
                    "cve_pub":        cve_pub,
                    "lag_anssi_days": lag_anssi,
                    # ─── EPSS ───────────────────────────────────
                    "epss_score":     epss_score,
                    "epss_percentile":epss_percentile,
                    # ─── produit ────────────────────────────────
                    "vendor":     vendor_raw,
                    "vendor_std": vendor_std,
                    "produit":    produit,
                    "versions":   versions,
                })
            )

    # ───────── DataFrame final & conversions ────────────────────────────
    df = pd.DataFrame(rows).drop_duplicates()

    # dates
    df["date"]      = pd.to_datetime(df["date"].str[:10],      errors="coerce")
    df["closed_at"] = pd.to_datetime(df["closed_at"].str[:10], errors="coerce")
    df["cve_pub"]   = pd.to_datetime(df["cve_pub"],             errors="coerce")

    today = pd.Timestamp("now").normalize()
    df["closed_at"] = df["closed_at"].fillna(today)

    # days_open
    df["days_open"] = (df["closed_at"] - df["date"]).dt.days

    # types numériques
    num_cols = ["cvss_score", "epss_score", "epss_percentile", "lag_anssi_days"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
