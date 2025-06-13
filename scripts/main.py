from src.extract_cves import extract_all_entries
from src.consolidate import build_dataframe
from src.alerts import send_critical_alerts

def main():
    print("Lancement du pipeline ANSSI simplifié...")

    # Étape 1 : Extraire les bulletins et les CVE
    all_entries = extract_all_entries(
        avis_dir="data/raw/Avis",
        alertes_dir="data/raw/alertes"
    )

    # Étape 2 : Créer le DataFrame enrichi à partir des fichiers MITRE + EPSS
    df = build_dataframe(
        entries=all_entries,
        mitre_dir="data/raw/mitre",
        epss_dir="data/raw/first"
    )

    # Étape 3 : Envoi d’alertes critiques
    send_critical_alerts(df)

if __name__ == "__main__":
    main()