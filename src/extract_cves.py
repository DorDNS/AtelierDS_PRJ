# Parcourir les fichiers JSON du dossier Avis/ et alertes/ pour en extraire les CVE + métadonnées

import os
import json
from datetime import datetime

def parse_file(path, type_bulletin):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    # Champs de base
    id_anssi = data.get("reference")
    titre = data.get("title", "")
    revisions = data.get("revisions", [])
    date = revisions[0]["revision_date"][:10] if revisions else None
    lien = None

    # Lien vers le bulletin d'origine
    for link in data.get("links", []):
        if "cert.ssi.gouv.fr" in link["url"]:
            lien = link["url"]
            break

    # Extraire les CVE
    cves = [cve["name"] for cve in data.get("cves", [])]

    entries = []
    for cve in cves:
        entries.append({
            "id_anssi": id_anssi,
            "type": type_bulletin,
            "titre": titre,
            "date": date,
            "lien": lien,
            "cve": cve
        })

    return entries

def extract_all_entries(avis_dir="data/raw/Avis", alertes_dir="data/raw/alertes"):
    all_entries = []

    # Parcourir les fichiers d'avis
    for filename in os.listdir(avis_dir):
        path = os.path.join(avis_dir, filename)
        if os.path.isfile(path) and not filename.startswith('.'):
            all_entries.extend(parse_file(path, "avis"))

    # Parcourir les fichiers d'alertes
    for filename in os.listdir(alertes_dir):
        path = os.path.join(alertes_dir, filename)
        if os.path.isfile(path) and not filename.startswith('.'):
            all_entries.extend(parse_file(path, "alerte"))

    return all_entries
