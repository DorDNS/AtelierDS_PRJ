# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# --- CONFIGURATION ---
load_dotenv()
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
EMAIL_SENDER = "burkovic.matthieu@gmail.com"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- UTILISATEUR (Simulé en mémoire) ---
user_preferences = {
    "email": "matthieu.burkovic@efrei.net",
    "produits": ["Windows", "Apache", "Linux"],
    "cvss": 8.0,
    "epss": 0.5
}

# --- CHARGEMENT ET FILTRAGE ---
def charger_et_filtrer_donnees(csv_path, produits, seuil_cvss, seuil_epss):
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        il_y_a_2_jours = datetime.now() - timedelta(days=2)
        df_recent = df[df["date"] >= il_y_a_2_jours]
        df_filtre = df_recent[
            (df_recent["cvss_score"] >= seuil_cvss) &
            (df_recent["epss_score"] >= seuil_epss) &
            (df_recent["produit"].isin(produits))
        ]
        return df_filtre
    except Exception as e:
        print(f"Erreur de chargement CSV : {e}")
        return pd.DataFrame()

# --- EMAIL ---
def creer_message_email(df_filtre):
    if df_filtre.empty:
        return "Aucune alerte critique récente."
    message = "🛡️ Alertes critiques récentes :\n\n"
    for _, row in df_filtre.iterrows():
        message += f"🔒 CVE: {row['cve']} ({row['cvss_sev']})\n"
        message += f"Produit: {row['produit']} | CVSS: {row['cvss_score']} | EPSS: {row['epss_score']}\n"
        message += f"{row['titre']}\n"
        message += f"{row['description'][:200]}...\n"
        message += f"{row['date'].strftime('%Y-%m-%d')}\n"
        message += f"Lien: {row['lien']}\n\n"
    return message

def envoyer_email(subject, body, receiver):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

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
    df = charger_et_filtrer_donnees("donnees_vuln.csv", user_preferences['produits'], user_preferences['cvss'], user_preferences['epss'])
    if not df.empty:
        message = creer_message_email(df)
        envoyer_email("Alertes de sécurité critiques", message, user_preferences['email'])

scheduler = BackgroundScheduler()
scheduler.add_job(tache_programmee, 'interval', hours=1)
scheduler.start()

# --- ROUTES WEB ---
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    df = charger_et_filtrer_donnees("donnees_vuln.csv", user_preferences['produits'], user_preferences['cvss'], user_preferences['epss'])
    return templates.TemplateResponse("dashboard.html", {"request": request, "df": df.to_dict(orient="records")})
