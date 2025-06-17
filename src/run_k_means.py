#!/usr/bin/env python3
"""
Script d'exécution de l'analyse K-Means
Auteur: Assistant IA
Date: 2025

Ce script lance une analyse K-Means complète avec:
- Détermination automatique du nombre optimal de clusters
- Visualisations des résultats
- Analyse des insights pour système de notifications
"""

import sys
import os
from datetime import datetime
import traceback

# Ajout du chemin src au PYTHONPATH pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from k_means import KMeansAnalyzer
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que le fichier k_means.py est dans le dossier src/")
    sys.exit(1)


def print_header():
    """Affiche l'en-tête du programme"""
    print("=" * 80)
    print("🔬 ANALYSE K-MEANS AUTOMATISÉE")
    print("=" * 80)
    print(f"📅 Exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Objectif: Détermination automatique du nombre optimal de clusters")
    print("📊 Données: final_dataset.csv")
    print("=" * 80)


def print_separator():
    """Affiche un séparateur visuel"""
    print("\n" + "─" * 80 + "\n")


def print_insights_summary(insights):
    """
    Affiche un résumé des insights pour le système de notifications

    Args:
        insights (dict): Dictionnaire contenant les insights
    """
    print("💡 INSIGHTS POUR LE SYSTÈME DE NOTIFICATIONS")
    print("=" * 60)

    print(f"\n🎯 Nombre optimal de clusters identifiés: {insights['optimal_clusters']}")

    print(f"\n📋 Profils des clusters:")
    for cluster_id, profile in insights['cluster_profiles'].items():
        print(f"\n    📌 Cluster {cluster_id}:")
        print(f"      - Taille: {profile['size']} points ({profile['percentage']:.1f}% du total)")
        print(f"      - Caractéristiques principales:")

        # Affichage des 3 caractéristiques les plus importantes
        sorted_chars = sorted(profile['characteristics'].items(),
                              key=lambda x: abs(x[1]), reverse=True)[:3]

        for feature, value in sorted_chars:
            print(f"         • {feature}: {value:.3f}")

    print(f"\n🚀 Recommandations stratégiques:")
    for i, recommendation in enumerate(insights['recommendations'], 1):
        print(f"   {i}. {recommendation}")

    print_separator()


def print_model_performance(results):
    """
    Affiche les performances du modèle

    Args:
        results (dict): Résultats de l'analyse
    """
    print("⚡ PERFORMANCES DU MODÈLE")
    print("=" * 50)

    print(f"🤖 Modèle K-Means entraîné avec K = {results['optimal_k']}")
    print(f"📊 Nombre total de points: {len(results['cluster_labels'])}")

    # Distribution des clusters
    from collections import Counter
    cluster_distribution = Counter(results['cluster_labels'])

    print(f"\n📈 Distribution des clusters:")
    for cluster_id in sorted(cluster_distribution.keys()):
        count = cluster_distribution[cluster_id]
        percentage = (count / len(results['cluster_labels'])) * 100
        print(f"   - Cluster {cluster_id}: {count:4d} points ({percentage:5.1f}%)")

    print_separator()


def print_next_steps():
    """Affiche les prochaines étapes recommandées"""
    print("🗺️  PROCHAINES ÉTAPES RECOMMANDÉES")
    print("=" * 50)

    next_steps = [
        "Analyser les caractéristiques spécifiques de chaque cluster",
        "Définir des stratégies de notification personnalisées par cluster",
        "Implémenter un système de scoring pour prioriser les notifications",
        "Tester l'efficacité des notifications sur un échantillon de chaque cluster",
        "Mettre en place un monitoring de la performance des clusters dans le temps",
        "Considérer une re-segmentation périodique pour s'adapter aux évolutions"
    ]

    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")

    print_separator()


def print_technical_summary(results):
    """
    Affiche un résumé technique de l'analyse

    Args:
        results (dict): Résultats de l'analyse
    """
    print("🔧 RÉSUMÉ TECHNIQUE")
    print("=" * 40)

    print(f"📏 Méthodes d'évaluation utilisées:")
    for method, data in results['metrics'].items():
        optimal_k = data['optimal_k']
        print(f"   - {method.replace('_', ' ').title()}: K = {optimal_k}")

    print(f"\n🎯 Consensus: K = {results['optimal_k']}")
    print(f"⚙️ Algorithme: K-Means avec initialisation aléatoire contrôlée")
    print(f"🔄 Préprocessing: Standardisation des données")
    print(f"🎲 Graine aléatoire: 42 (reproductibilité)")

    print_separator()


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ DÉPENDANCES MANQUANTES:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\n💡 Installez-les avec: pip install {' '.join(missing_packages)}")
        return False

    return True


def find_data_file():
    """Trouve le fichier de données dans différents emplacements possibles"""
    possible_paths = [
        'data/processed/final_dataset.csv',
        '../data/processed/final_dataset.csv',
        'final_dataset.csv',
        'data/final_dataset.csv'
    ]

    print("🔍 Recherche du fichier de données...")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Fichier trouvé: {path}")
            return path
        else:
            print(f"❌ Non trouvé: {path}")

    return None


def print_system_info():
    """Affiche les informations système pour le débogage"""
    print("🖥️  INFORMATIONS SYSTÈME")
    print("=" * 40)
    print(f"   Python: {sys.version}")
    print(f"   Répertoire de travail: {os.getcwd()}")
    print(f"   Système: {os.name}")

    # Vérification des dépendances
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print(f"   Pandas: {pd.__version__}")
        print(f"   NumPy: {np.__version__}")
        print(f"   Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("   ⚠️ Certaines dépendances ne sont pas disponibles")


def main():
    """Fonction principale d'exécution"""
    try:
        # Affichage de l'en-tête
        print_header()

        # Recherche du fichier de données
        data_path = find_data_file()
        if data_path is None:
            print("\n❌ ERREUR: Fichier de données introuvable!")
            print("   Vérifiez que 'final_dataset.csv' existe dans l'un de ces emplacements:")
            print("   - data/processed/final_dataset.csv")
            print("   - data/final_dataset.csv")
            print("   - final_dataset.csv")
            return None

        # Initialisation de l'analyseur avec le bon chemin
        print(f"\n🚀 Initialisation de l'analyseur K-Means...")
        analyzer = KMeansAnalyzer(data_path=data_path)

        print_separator()

        # Lancement de l'analyse complète
        print("🔄 Lancement de l'analyse complète...")
        results = analyzer.run_complete_analysis()

        if results is None:
            print("❌ Échec de l'analyse. Vérifiez vos données et réessayez.")
            return None

        return results

    except ImportError as e:
        print(f"\n❌ ERREUR: Problème d'importation des dépendances")
        print(f"   Détails: {str(e)}")
        print(f"   Solution: Installez les dépendances requises avec:")
        print(f"   pip install pandas numpy matplotlib seaborn scikit-learn")
        return None

    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {str(e)}")
        print(f"\n📋 Trace complète de l'erreur:")
        traceback.print_exc()
        print(f"\n💡 Suggestions de dépannage:")
        print(f"   1. Vérifiez que vos données sont au bon format")
        print(f"   2. Assurez-vous d'avoir suffisamment de mémoire disponible")
        print(f"   3. Vérifiez les permissions d'accès aux fichiers")
        return None


if __name__ == "__main__":
    """Point d'entrée du script"""

    try:
        # Vérifications préliminaires optionnelles (pour debug)
        if len(sys.argv) > 1 and sys.argv[1] == "--debug":
            print_system_info()
            print_separator()

            print("🔍 VÉRIFICATIONS PRÉLIMINAIRES")
            print("=" * 40)

            if not check_dependencies():
                sys.exit(1)
            print("✅ Toutes les dépendances sont installées")

            print_separator()

        # Exécution principale
        results = main()

        # Vérification que l'analyse a réussi
        if results is None:
            print("\n❌ L'analyse n'a pas pu être complétée.")
            sys.exit(1)

        print_separator()

        # Affichage des résultats
        print_model_performance(results)
        print_insights_summary(results['insights'])
        print_technical_summary(results)
        print_next_steps()

        # Message de fin
        print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("=" * 80)
        print("📊 Les graphiques ont été générés et affichés.")
        print("💾 Le modèle est prêt à être utilisé pour la segmentation.")
        print("🚀 Vous pouvez maintenant implémenter votre système de notifications!")
        print("=" * 80)

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n❌ ERREUR: Fichier de données introuvable")
        print(f"   Détails: {str(e)}")
        print(f"   Solution: Vérifiez que le fichier 'final_dataset.csv' existe")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {str(e)}")
        sys.exit(1)