# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Form
from fastapi.responses import RedirectResponse
from pathlib import Path



# --- CONFIGURATION ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
CSV_PATH = "data/processed/final_dataset.csv"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
EMAIL_SENDER = "votre_gmail@gmail.com" # Remplacez par votre adresse email

app = FastAPI()
templates = Jinja2Templates(directory="templates")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- UTILISATEUR (Simulé en mémoire) ---
user_preferences = {
    "email": "votre_mail@gmail.com", # Remplacez par l'adresse email du destinataire,
    "cvss": 7.0,
    "epss": 0.5,
    "vendor": "Tous"
}

# --- CHARGEMENT ET FILTRAGE ---
def charger_et_filtrer_donnees(csv_path, seuil_cvss, seuil_epss, vendor=None):
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        date_limite = datetime(datetime.now().year, 1, 1)
        df_recent = df[df["date"] >= date_limite]
        df_filtre = df_recent[
            (df_recent["cvss_score"] >= seuil_cvss) &
            (df_recent["epss_score"] >= seuil_epss)
        ]
        if vendor and vendor != "Tous":
            df_filtre = df_filtre[df_filtre["vendor"] == vendor]
        return df_filtre
    except Exception as e:
        print(f"Erreur de chargement CSV : {e}")
        return pd.DataFrame()

# --- EMAIL ---
def creer_message_email(df_filtre):
    if df_filtre.empty:
        return "Aucune alerte critique récente."
    message = "<h2>🛡️ Alertes critiques :</h2><ul>"
    for _, row in df_filtre.iterrows():
        description = row['description']
        if not isinstance(description, str):
            description = ""
        message += "<li>"
        message += f"<strong>CVE:</strong> {row['cve']} ({row['cvss_sev']})<br>"
        message += f"<strong>Produit:</strong> {row['produit']} | CVSS: {row['cvss_score']} | EPSS: {row['epss_score']}<br>"
        message += f"<strong>{row['titre']}</strong><br>"
        message += f"{description[:200]}...<br>"
        message += f"<em>{row['date'].strftime('%Y-%m-%d')}</em><br>"
        message += f"<a href='{row['lien']}' target='_blank'>Lien vers la vulnérabilité</a>"
        message += "</li><br>"
    message += "</ul>"
    return message

def envoyer_email(subject, body, receiver):
    """
    Envoie un email simple avec sujet et corps.
    Utilise les paramètres globaux SMTP_SERVER, SMTP_PORT, EMAIL_SENDER et EMAIL_PASSWORD.
    """
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = receiver
    msg['Subject'] = subject

    # Detecte si le body est HTML ou texte simple (optionnel, ici on met html)
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email envoyé avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email: {e}")

# --- PLANIFICATEUR ---
def tache_programmee():
    df = charger_et_filtrer_donnees(
        "data/processed/final_dataset.csv",
        user_preferences['cvss'],
        user_preferences['epss']
    )
    if not df.empty:
        message = creer_message_email(df)
        envoyer_email("Alertes de sécurité critiques", message, user_preferences['email'])

scheduler = BackgroundScheduler()
scheduler.add_job(tache_programmee, 'interval', hours=1)
scheduler.start()

# --- ROUTES WEB ---
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    df = charger_et_filtrer_donnees(
        CSV_PATH,
        user_preferences['cvss'],
        user_preferences['epss'],
        user_preferences.get('vendor') 
    )
    return templates.TemplateResponse("dashboard.html", {"request": request, "df": df.to_dict(orient="records")})




@app.post("/send-alert-mail")
async def send_alert_mail(cve: str = Form(...)):
    df = charger_et_filtrer_donnees(
        "data/processed/final_dataset.csv",
        user_preferences['cvss'],
        user_preferences['epss']
    )
    alert = df[df['cve'] == cve]
    if alert.empty:
        return RedirectResponse(url="/", status_code=303)

    message = creer_message_email(alert)
    envoyer_email("Alerte de sécurité critique - " + cve, message, user_preferences['email'])

    return RedirectResponse(url="/", status_code=303)

@app.get("/preferences")
async def get_preferences(request: Request):
    vendors = charger_tous_les_vendors(CSV_PATH) 
    vendors = sorted(vendors)
    if "" not in vendors:
        vendors = [""] + vendors

    return templates.TemplateResponse(
        "preferences.html",
        {
            "request": request,
            "email": user_preferences.get("email", ""),
            "cvss": user_preferences.get("cvss", 7.0),
            "epss": user_preferences.get("epss", 0.5),
            "selected_vendor": user_preferences.get("vendor", ""),
            "vendors": vendors,
            "active_tab": "preferences"
        }
    )

@app.post("/preferences")
async def update_preferences(
    email: str = Form(...),
    cvss: float = Form(...),
    epss: float = Form(...),
    vendor: str = Form("")  
):
    # Met à jour les préférences en mémoire
    user_preferences["email"] = email
    user_preferences["cvss"] = cvss
    user_preferences["epss"] = epss
    user_preferences["vendor"] = vendor  

    return RedirectResponse(url="/preferences", status_code=303)


def charger_tous_les_vendors(csv_path):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        
        vendors = df['vendor'].dropna().map(str.strip).unique()
        vendors = sorted(vendors)
        return vendors
    except Exception as e:
        print(f"Erreur chargement vendors : {e}")
        return []

