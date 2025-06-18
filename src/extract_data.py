from __future__ import annotations
import concurrent.futures as cf, datetime as dt, functools, re, time
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse
import csv

import pandas as pd
import requests
from bs4 import BeautifulSoup

# chemins & sortie CSV
ROOT_DIR   = Path(__file__).resolve().parent.parent            
OUT_DIR    = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)                     
CSV_PATH   = OUT_DIR / "final_dataset.csv"                     

# paramètres scraping
BASE_URLS = {
    "Alerte": "https://cert.ssi.gouv.fr/alerte",
    "Avis":   "https://cert.ssi.gouv.fr/avis",
}
ROOT, TIMEOUT = "https://cert.ssi.gouv.fr", 6
HEADERS       = {"User-Agent": "CERTFR Scraper // EFREI 2025"}
PAGE_WORKERS, MAX_WORKERS = 10, 20
SINCE_DATE = pd.Timestamp("2020-01-01") # historique ≥ 2020

CVE_RX     = re.compile(r"CVE-\d{4}-\d{4,7}")
CVSS_BANDS = [
    (0, 4,  "Faible"),
    (4, 7,  "Moyenne"),
    (7, 9,  "Élevée"),
    (9, 11, "Critique"),
]

# normalisation vendor
_VENDOR_MAP: Dict[str, str] = {
    "microsoft corporation": "Microsoft",
    "microsoft corp.":       "Microsoft",
    "microsoft":             "Microsoft",
    "google llc":            "Google",
    "google":                "Google",
    "oracle corporation":    "Oracle",
    "oracle":                "Oracle",
    "international business machines corporation": "IBM",
    "ibm":                   "IBM",
}
normalize_vendor = lambda v: None if v is None else _VENDOR_MAP.get(v.lower(), v)

# sessions HTTP 
SESSION = requests.Session()     # pour HTML + APIs MITRE/EPSS
SESSION.headers.update(HEADERS)

PLAIN   = requests.Session()     # pour /json/ des bulletins
PLAIN.headers.update(HEADERS)

# helpers
def page_url(sec:str,n:int): return f"{BASE_URLS[sec]}/" if n==1 else f"{BASE_URLS[sec]}/page/{n}/"
single_line = lambda t: None if t is None else re.sub(r"\s+"," ",t).strip()
def cvss_band(s): 
    return next((lab for lo,hi,lab in CVSS_BANDS if s is not None and lo<=s<hi), None)

def safe_get(d:Any, keys:List, default=None):
    for k in keys:
        try: d = d[k]
        except (KeyError,IndexError,TypeError): return default
    return d

# date extraction robuste
def extract_date(j:dict) -> pd.Timestamp|None:
    for key in ("initial_release_date","modified_release_date","last_revision_date"):
        if key in j and j[key]:
            return pd.to_datetime(j[key], errors="coerce")
    rev = safe_get(j, ["revisions",0,"revision_date"])
    return pd.to_datetime(rev, errors="coerce")

def extract_closed(j:dict) -> pd.Timestamp|None:
    if j.get("closed_at"):
        return pd.to_datetime(j["closed_at"], errors="coerce")
    rev_last = safe_get(j, ["revisions",-1,"revision_date"])
    return pd.to_datetime(rev_last, errors="coerce")

# pagination HTM
def _fetch_page(sec:str,n:int):
    r = SESSION.get(page_url(sec,n),timeout=TIMEOUT)
    return n,r.status_code,r.text

def iter_bulletin_urls(sec:str):
    p=1
    with cf.ThreadPoolExecutor(PAGE_WORKERS) as pool:
        while True:
            futs=[pool.submit(_fetch_page,sec,n) for n in range(p,p+PAGE_WORKERS)]
            stop=False
            for f in cf.as_completed(futs):
                n,st,html=f.result()
                if st==404: stop=True; continue
                soup=BeautifulSoup(html,"lxml")
                links=[urljoin(ROOT,a["href"]).rstrip("/") for a in soup.select("h3 > a[href]")]
                print(f"[{sec}] page {n:>3} — {len(links):>2} bulletins", flush=True)
                yield from links
                if not soup.find("a",string=re.compile("Suivant")): stop=True
            if stop: break
            p+=PAGE_WORKERS

# enrichissement MITRE / EPSS
@functools.lru_cache(maxsize=None)
def enrich_cve(cve:str):
    try:
        mitre = SESSION.get(f"https://cveawg.mitre.org/api/cve/{cve}", timeout=TIMEOUT).json()
    except Exception:
        return (None,None,None,None,None,None,0,None,[("n/a","n/a","n/a")],None)

    cna = mitre.get("containers",{}).get("cna",{})
    desc = single_line(next((d["value"] for d in cna.get("descriptions",[]) if d.get("lang")=="en"), None))

    cvss_score=cvss_sev=None
    for m in cna.get("metrics",[]):
        for key in ("cvssV3_1","cvssV3_0","cvssV3","cvssV2"):
            if key in m:
                blk=m[key]; cvss_score=blk.get("baseScore"); cvss_sev=blk.get("baseSeverity"); break
        if cvss_score is not None: break

    pt=safe_get(cna,["problemTypes",0,"descriptions"],[])
    cwe=cwe_desc=None
    if pt:
        cwe=pt[0].get("cweId") or pt[0].get("description")
        cwe_desc=single_line(pt[0].get("description"))

    n_refs=len(cna.get("references",[]))
    cve_pub=safe_get(mitre,["cveMetadata","datePublished"])
    cve_pub=cve_pub[:10] if cve_pub else None

    products=[(p.get("vendor","n/a"),p.get("product","n/a"),
               ", ".join(v.get("version","") if isinstance(v,dict) else str(v)
                         for v in p.get("versions",[])) or "n/a")
              for p in cna.get("affected",[])] or [("n/a","n/a","n/a")]

    epss_score=epss_pct=None
    try:
        eps=SESSION.get(f"https://api.first.org/data/v1/epss?cve={cve}",timeout=TIMEOUT).json()
        if eps.get("data"):
            ep=eps["data"][0]; epss_score=float(ep["epss"]); epss_pct=float(ep["percentile"])
    except Exception: pass

    return (cvss_score,cvss_sev,cwe,cwe_desc,
            epss_score,epss_pct,n_refs,cve_pub,products,desc)

# pipeline
def build_dataframe(sections=("Alerte","Avis"))->pd.DataFrame:
    t0=time.perf_counter()
    urls=[u for s in sections for u in iter_bulletin_urls(s)]
    print(f"\n→ {len(urls):,} bulletins collectés ({time.perf_counter()-t0:.1f}s)\n")

    # JSON
    def fetch_json(u:str):
        try:
            r=PLAIN.get(u+"/json/",timeout=TIMEOUT)
            return u,(r.json() if r.status_code==200 else None)
        except requests.RequestException:
            return u,None
    print("◆ Téléchargement des JSON …")
    with cf.ThreadPoolExecutor(MAX_WORKERS) as pool:
        json_map=dict(pool.map(fetch_json, urls))
    print("   ✓ JSON terminés\n")

    # parse + filtre
    meta,cve_set=[],set()
    for url,j in json_map.items():
        if not j: continue
        date = extract_date(j)
        if pd.isna(date) or date < SINCE_DATE:  # filtre 2020 →
            continue

        sec   = "Alerte" if "/alerte/" in url else "Avis"
        bid   = j.get("id") or Path(urlparse(url).path).parts[-1]
        title = j.get("title","")
        cves  = [c["name"] for c in j.get("cves",[])] or CVE_RX.findall(str(j)) or [None]
        cert_desc = single_line(j.get("description",""))
        closed_at = extract_closed(j)
        n_revisions = len(j.get("revisions", []))

        meta.append((sec, bid, title, date, url, cves, cert_desc, closed_at, n_revisions))
        cve_set.update([c for c in cves if c])

    print(f"→ {len(meta):,} bulletins ≥2020   ·   {len(cve_set):,} CVE uniques\n")

    if not meta:      # rien à traiter -> DataFrame vide
        print("Aucun bulletin dans la plage demandée.")
        return pd.DataFrame()

    # enrichissement
    print("◆ Enrichissement MITRE/EPSS …")
    with cf.ThreadPoolExecutor(MAX_WORKERS) as pool:
        futs={pool.submit(enrich_cve,c):c for c in cve_set}
        for i,_ in enumerate(cf.as_completed(futs),1):
            if i%2_000==0 or i==len(futs):
                print(f"    {i:,}/{len(futs):,} enrichis", flush=True)
    print("   ✓ enrichissement terminé\n")

    # consolidation
    today=pd.Timestamp("now").normalize()
    rows=[]
    for sec,bid,title,date,url,cves,cert_desc,closed_at,n_revisions in meta:
        closed_at=closed_at or today
        for cve in cves:
            (cvss_score,cvss_sev,cwe,cwe_desc,
             epss_score,epss_pct,n_refs,cve_pub,products,mitre_desc)=(
                 enrich_cve(cve) if cve else
                 (None,None,None,None,None,None,0,None,[("n/a","n/a","n/a")],None))

            lag=None
            if cve_pub:
                try: lag=(date - pd.to_datetime(cve_pub)).days
                except Exception: pass

            description=mitre_desc or cert_desc
            for vendor_raw,prod,vers in products:
                rows.append({
                    "id_anssi":bid, "type":sec.lower(), "titre":title,
                    "date":date.date(), "closed_at":closed_at.date() if pd.notna(closed_at) else None,
                    "lien":url, "cve":cve, "description":description, "n_revisions":n_revisions,
                    "cvss_score":cvss_score, "cvss_sev":cvss_sev or cvss_band(cvss_score),
                    "cwe":cwe, "cwe_description":cwe_desc,
                    "n_cve_refs":n_refs, "cve_pub":cve_pub, "lag_anssi_days":lag,
                    "epss_score":epss_score, "epss_percentile":epss_pct,
                    "vendor":vendor_raw, "vendor_std":normalize_vendor(vendor_raw),
                    "produit":prod, "versions":vers,
                })

    df=pd.DataFrame(rows).drop_duplicates()
    df["date"]=pd.to_datetime(df["date"])
    df["closed_at"]=pd.to_datetime(df["closed_at"]).fillna(today)
    df["cve_pub"]=pd.to_datetime(df["cve_pub"], errors="coerce")
    df["days_open"]=(df["closed_at"]-df["date"]).dt.days
    for col in ["cvss_score","epss_score","epss_percentile","lag_anssi_days"]:
        df[col]=pd.to_numeric(df[col], errors="coerce")

    print(f"→ DataFrame final : {len(df):,} lignes "
          f"(total {time.perf_counter()-t0:.1f}s)\n")
    return df

def main():
    t0 = time.perf_counter()
    df = build_dataframe()
    if not df.empty:
        df.to_csv(
            CSV_PATH,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_NONNUMERIC,
            quotechar='"'
        )
        print(f"{len(df):,} lignes écrites dans {CSV_PATH.name} "
              f"( {time.perf_counter()-t0:.1f} s )")

if __name__ == "__main__":
    main()