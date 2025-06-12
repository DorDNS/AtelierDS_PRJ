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