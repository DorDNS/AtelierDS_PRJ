from src.extract_rss import get_rss_entries
from src.extract_cves import extract_cves_from_json
from src.enrich_cves import enrich_cve_data
from src.consolidate import build_dataframe
from src.alerts import send_critical_alerts

def main():
    print("Lancement du pipeline ANSSI...")
    rss_entries = get_rss_entries()
    all_cve_data = extract_cves_from_json(rss_entries)
    enriched = enrich_cve_data(all_cve_data)
    df = build_dataframe(enriched)
    send_critical_alerts(df)

if __name__ == "__main__":
    main()