# Génération et envoi des emails

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
EMAIL_SENDER = "thomas.fischer67270@gmail.com"
EMAIL_RECEIVER = "thomas.fischer@efrei.net"

def filtrer_alertes(df, produits_cibles, seuil_cvss=7.0, seuil_epss=0.2):
    """
    Filtre le DataFrame pour ne garder que les vulnérabilités critiques/à risque élevé
    et qui concernent les produits cibles.
    """
    filtered = df[
        (df["Score CVSS"] >= seuil_cvss) &
        (df["Score EPSS"] >= seuil_epss) &
        (df["Produit"].isin(produits_cibles))
    ]
    return filtered

def creer_message_email(df_filtre):
    """
    Crée le contenu textuel du mail à partir du DataFrame filtré.
    """
    if df_filtre.empty:
        return "Aucune alerte critique récente à signaler pour vos produits."

    message = "🛡️ Alertes de vulnérabilités critiques détectées ces deux derniers jours :\n\n"
    for _, row in df_filtre.iterrows():
        message += f"🔒 CVE : {row['cve']} ({row['cvss_sev']})\n"
        message += f"Produit : {row['produit']} | Score CVSS : {row['cvss_score']} | Score EPSS : {row['epss_score']}\n"
        message += f"Titre : {row['titre']}\n"
        message += f"Description : {row['description'][:200]}...\n"
        message += f"Date : {row['date'].strftime('%Y-%m-%d')}\n"
        message += f"Lien : {row['lien']}\n"
        message += "\n"
    return message

def envoyer_email(subject, body, sender, receiver, smtp_server, smtp_port, password):
    """
    Envoie un email simple avec sujet et corps.
    """
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("Email envoyé avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email: {e}")


def charger_et_filtrer_donnees(csv_path, produits_cibles, seuil_cvss=7.0):
    """
    Charge le CSV, filtre les alertes critiques récentes et les produits cibles.
    """
    # Chargement du CSV avec parsing de la colonne date
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Filtrer sur les deux derniers jours
    maintenant = datetime.now()
    il_y_a_2_jours = maintenant - timedelta(days=64)
    df_recent = df[df["date"] >= il_y_a_2_jours]

    # Filtrer les vulnérabilités critiques et produits cibles
    df_filtre = df_recent[
        (df_recent["cvss_score"] >= seuil_cvss) &
        (df_recent["produit"].isin(produits_cibles))
    ]

    return df_filtre