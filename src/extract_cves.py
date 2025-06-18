import os
import json
from datetime import datetime
from typing import List, Dict


def parse_file(path: str, type_bulletin: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # champs ANSSI
    id_anssi   = data.get("reference")
    titre      = data.get("title", "")
    revisions  = data.get("revisions", [])
    n_revisions = len(revisions)
    date       = revisions[0]["revision_date"][:10] if revisions else None
    closed_at  = data.get("closed_at")              # ★ nouveau
    lien       = None

    for link in data.get("links", []):
        if "cert.ssi.gouv.fr" in link.get("url", ""):
            lien = link["url"]
            break

    # CVE
    cves = [cve["name"] for cve in data.get("cves", [])]

    entries = []
    for cve in cves:
        entries.append(
            {
                "id_anssi":  id_anssi,
                "type":      type_bulletin,
                "titre":     titre,
                "n_revisions": n_revisions,
                "date":      date,
                "closed_at": closed_at,   # ★
                "lien":      lien,
                "cve":       cve,
            }
        )
    return entries


def extract_all_entries(
    avis_dir: str = "data/raw/Avis",
    alertes_dir: str = "data/raw/alertes",
) -> List[Dict]:
    all_entries: List[Dict] = []

    # Avis
    for filename in os.listdir(avis_dir):
        path = os.path.join(avis_dir, filename)
        if os.path.isfile(path) and not filename.startswith("."):
            all_entries.extend(parse_file(path, "avis"))

    # Alertes
    for filename in os.listdir(alertes_dir):
        path = os.path.join(alertes_dir, filename)
        if os.path.isfile(path) and not filename.startswith("."):
            all_entries.extend(parse_file(path, "alerte"))

    return all_entries