# Projet - Analyse des Alertes ANSSI & Enrichissement CVE

Ce projet vise à extraire les avis et alertes ANSSI, enrichir les CVE avec les API MITRE et EPSS, analyser les vulnérabilités et générer des alertes personnalisées.

## Structure
- `src/` : Code source
- `notebooks/` : Analyse, visualisations et Machine Learning
- `data/` : Données brutes et traitées
- `scripts/` : Point d’entrée du projet

## Installation
```bash
pip install -r requirements.txt
```

## Lancement

```bash
python scripts/main.py
```

## Données brutes (non incluses dans le repo)

Les dossiers `Avis/`, `alertes/`, `mitre/` et `first/` contiennent des centaines de fichiers JSON fournis par l'enseignant.

Pour alléger le dépôt GitHub, ces fichiers **ne sont pas versionnés** (voir `.gitignore`).

Merci de copier manuellement le dossier `data_pour_TD_final/` dans ce dossier `data/raw/` en respectant l’arborescence :

```
data/raw/
├── Avis/
├── alertes/
├── mitre/
└── first/
```


## Pour lancer le dashboard

Créer un dossier nommé "static" si il n'existe pas dans le projet (en-dehors de tout dossier)

télécharger : pip install python-multipart

Finalement : 
uvicorn src.dashboard_vuln_alerts:app --reload
