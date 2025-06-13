import os
import json
import pandas as pd

def safe_get(d, keys, default=None):
    """Accès sécurisé à une structure imbriquée."""
    for key in keys:
        try:
            d = d[key]
        except (KeyError, IndexError, TypeError):
            return default
    return d

def clean_row(row_dict):
    """Remplace None par 'n/a', nettoie la description, et convertit les listes en chaînes."""
    clean = {}
    for k, v in row_dict.items():
        if v is None:
            clean[k] = "n/a"
        elif isinstance(v, list):
            clean[k] = " | ".join(str(i) for i in v) if v else "n/a"
        elif k == "description":
            # supprime les retours à la ligne et tabulations, collapse des espaces
            clean[k] = " ".join(str(v).split())
        else:
            clean[k] = v
    return clean

def build_dataframe(entries,
                    mitre_dir="data/raw/mitre",
                    epss_dir="data/raw/first"):
    rows = []

    for entry in entries:
        cve = entry.get("cve")
        if not cve:
            continue

        # chemins vers les JSON MITRE et EPSS
        mitre_path = os.path.join(mitre_dir, cve)
        epss_path  = os.path.join(epss_dir,  cve)

        # lien dynamique
        lien = f"https://www.cert.ssi.gouv.fr/{'alerte' if entry.get('type')=='alerte' else 'avis'}/{entry.get('id_anssi')}/"

        # --- chargement MITRE ---
        if not os.path.exists(mitre_path):
            continue
        try:
            with open(mitre_path, encoding="utf-8") as f:
                mitre_data = json.load(f)
        except Exception:
            continue

        cna = safe_get(mitre_data, ["containers","cna"], {})

        # --- description anglais ---
        description = next(
            (d["value"] for d in cna.get("descriptions", [])
             if d.get("lang") == "en"),
            None
        )

        # --- CVSS ---
        cvss_score    = None
        base_severity = None
        for metric in cna.get("metrics", []):
            for key in ("cvssV3_1","cvssV3_0","cvssV3"):
                if key in metric:
                    cvss_score    = metric[key].get("baseScore")
                    base_severity = metric[key].get("baseSeverity")
                    break
            if cvss_score is not None:
                break

        # --- CWE ---
        cwe = cwe_description = None
        cwe_list = safe_get(cna, ["problemTypes",0,"descriptions"], [])
        if cwe_list:
            cwe             = cwe_list[0].get("cweId") or cwe_list[0].get("description")
            cwe_description = cwe_list[0].get("description")

        # --- EPSS ---
        epss_score = None
        if os.path.exists(epss_path):
            try:
                with open(epss_path, encoding="utf-8") as f:
                    epss = json.load(f)
                raw = safe_get(epss, ["data",0,"epss"])
                if raw is not None:
                    epss_score = float(raw)
            except Exception:
                pass

        # --- produits affectés ---
        affected = cna.get("affected", [])
        if not affected:
            row = {
                "id_anssi":      entry.get("id_anssi"),
                "type":          entry.get("type"),
                "titre":         entry.get("titre"),
                "date":          entry.get("date"),
                "lien":          lien,
                "cve":           cve,
                "description":   description,
                "cvss_score":    cvss_score,
                "base_severity": base_severity,
                "cwe":           cwe,
                "cwe_description": cwe_description,
                "epss_score":    epss_score,
                "vendor":        "n/a",
                "produit":       "n/a",
                "versions":      ["n/a"],
            }
            rows.append(clean_row(row))
        else:
            for prod in affected:
                vendor   = prod.get("vendor", "n/a")
                produit  = prod.get("product", "n/a")
                versions = [v.get("version","n/a")
                            for v in prod.get("versions",[])
                            if v.get("status")=="affected"] or ["n/a"]

                row = {
                    "id_anssi":      entry.get("id_anssi"),
                    "type":          entry.get("type"),
                    "titre":         entry.get("titre"),
                    "date":          entry.get("date"),
                    "lien":          lien,
                    "cve":           cve,
                    "description":   description,
                    "cvss_score":    cvss_score,
                    "base_severity": base_severity,
                    "cwe":           cwe,
                    "cwe_description": cwe_description,
                    "epss_score":    epss_score,
                    "vendor":        vendor,
                    "produit":       produit,
                    "versions":      versions,
                }
                rows.append(clean_row(row))

    df = pd.DataFrame(rows).drop_duplicates()
    return df
