# Génération et envoi des emails

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
EMAIL_SENDER = "burkovic.matthieu@gmail.com"
EMAIL_RECEIVER = "matthieu.burkovic@efrei.net"

def filtrer_alertes(df, produits_cibles, seuil_cvss=7.0, seuil_epss=0.7):
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
    Crée le contenu HTML/textuel du mail à partir du DataFrame filtré.
    """
    if df_filtre.empty:
        return "Aucune alerte critique à signaler pour vos produits."

    message = "Alertes de vulnérabilités critiques pour vos produits:\n\n"
    for idx, row in df_filtre.iterrows():
        message += f"- {row['CVE']} ({row['Base Severity']}): {row['Description']}\n"
        message += f"  Produit : {row['Produit']}\n"
        message += f"  Score CVSS : {row['Score CVSS']} | Score EPSS : {row['Score EPSS']}\n"
        message += f"  Plus d'infos : {row['Lien']}\n\n"
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





def envoyer_email_test():
    sujet = "✅ Test d'envoi d'email avec Python"
    corps = "Bonjour,\n\nCeci est un test automatique de l'envoi d'email depuis un script Python avec SMTP Gmail.\n\n– Script terminé avec succès 🚀"

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = sujet
    msg.attach(MIMEText(corps, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email de test envoyé avec succès.")
    except Exception as e:
        print(f"❌ Échec de l'envoi de l'email : {e}")


envoyer_email_test()