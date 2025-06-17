import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class KMeansAnalyzer:
    """
    Classe pour l'analyse K-Means avec détermination automatique du nombre optimal de clusters
    """

    def __init__(self, data_path='data/processed/final_dataset.csv'):
        """
        Initialise l'analyseur K-Means

        Args:
            data_path (str): Chemin vers le fichier de données
        """
        self.data_path = data_path
        self.data = None
        self.scaled_data = None
        self.scaler = None
        self.optimal_k = None
        self.kmeans_model = None
        self.cluster_labels = None
        self.metrics_results = {}

    def load_and_preprocess_data(self):
        """
        Charge et préprocesse les données
        """
        try:
            # Chargement des données
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Données chargées: {self.data.shape[0]} lignes, {self.data.shape[1]} colonnes")

            # Affichage des informations sur les données
            print("\n Informations sur le dataset:")
            print(f"   - Colonnes: {list(self.data.columns)}")
            print(f"   - Types de données:\n{self.data.dtypes}")
            print(f"   - Valeurs manquantes:\n{self.data.isnull().sum()}")

            # Sélection des colonnes numériques uniquement
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data_numeric = self.data[numeric_columns].copy()

            # Suppression des valeurs manquantes
            self.data_numeric = self.data_numeric.dropna()

            # Standardisation des données
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.data_numeric)

            print(f"✓ Données préprocessées: {self.scaled_data.shape[0]} lignes, {self.scaled_data.shape[1]} features")
            print(f"   - Features utilisées: {list(numeric_columns)}")

            return True

        except Exception as e:
            print(f" Erreur lors du chargement des données: {str(e)}")
            return False

    def elbow_method(self, max_k=15):
        """
        Méthode du coude pour déterminer le nombre optimal de clusters

        Args:
            max_k (int): Nombre maximum de clusters à tester

        Returns:
            tuple: (liste des K, liste des inerties, K optimal estimé)
        """
        print("\n Application de la méthode du coude...")

        k_range = range(1, max_k + 1)
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)

        # Calcul du "coude" - recherche du point d'inflexion
        # Méthode des différences secondes
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_derivatives.append(inertias[i - 1] - 2 * inertias[i] + inertias[i + 1])

            # Le coude correspond au maximum de la dérivée seconde
            elbow_k = second_derivatives.index(max(second_derivatives)) + 2
        else:
            elbow_k = 3  # valeur par défaut

        self.metrics_results['elbow'] = {'k_values': list(k_range), 'inertias': inertias, 'optimal_k': elbow_k}

        print(f"   ✓ Méthode du coude suggère K = {elbow_k}")
        return k_range, inertias, elbow_k

    def silhouette_analysis(self, max_k=15):
        """
        Analyse par coefficient de silhouette

        Args:
            max_k (int): Nombre maximum de clusters à tester

        Returns:
            tuple: (liste des K, liste des scores, K optimal)
        """
        print("\n Analyse par coefficient de silhouette...")

        k_range = range(2, max_k + 1)  # Silhouette nécessite au moins 2 clusters
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]

        self.metrics_results['silhouette'] = {
            'k_values': list(k_range),
            'scores': silhouette_scores,
            'optimal_k': optimal_k
        }

        print(f"   ✓ Coefficient de silhouette suggère K = {optimal_k} (score: {max(silhouette_scores):.3f})")
        return k_range, silhouette_scores, optimal_k

    def calinski_harabasz_analysis(self, max_k=15):
        """
        Analyse par indice de Calinski-Harabasz

        Args:
            max_k (int): Nombre maximum de clusters à tester

        Returns:
            tuple: (liste des K, liste des scores, K optimal)
        """
        print("\n🔍 Analyse par indice de Calinski-Harabasz...")

        k_range = range(2, max_k + 1)
        ch_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            ch_score = calinski_harabasz_score(self.scaled_data, cluster_labels)
            ch_scores.append(ch_score)

        optimal_k = k_range[ch_scores.index(max(ch_scores))]

        self.metrics_results['calinski_harabasz'] = {
            'k_values': list(k_range),
            'scores': ch_scores,
            'optimal_k': optimal_k
        }

        print(f"   ✓ Indice de Calinski-Harabasz suggère K = {optimal_k} (score: {max(ch_scores):.2f})")
        return k_range, ch_scores, optimal_k

    def davies_bouldin_analysis(self, max_k=15):
        """
        Analyse par indice de Davies-Bouldin (plus faible = meilleur)

        Args:
            max_k (int): Nombre maximum de clusters à tester

        Returns:
            tuple: (liste des K, liste des scores, K optimal)
        """
        print("\n Analyse par indice de Davies-Bouldin...")

        k_range = range(2, max_k + 1)
        db_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            db_score = davies_bouldin_score(self.scaled_data, cluster_labels)
            db_scores.append(db_score)

        optimal_k = k_range[db_scores.index(min(db_scores))]

        self.metrics_results['davies_bouldin'] = {
            'k_values': list(k_range),
            'scores': db_scores,
            'optimal_k': optimal_k
        }

        print(f"   ✓ Indice de Davies-Bouldin suggère K = {optimal_k} (score: {min(db_scores):.3f})")
        return k_range, db_scores, optimal_k

    def determine_optimal_k(self):
        """
        Détermine le nombre optimal de clusters en combinant plusieurs méthodes

        Returns:
            int: Nombre optimal de clusters
        """
        print("\n Détermination du nombre optimal de clusters...")

        # Application de toutes les méthodes
        self.elbow_method()
        self.silhouette_analysis()
        self.calinski_harabasz_analysis()
        self.davies_bouldin_analysis()

        # Collecte des recommandations
        recommendations = []
        for method, results in self.metrics_results.items():
            recommendations.append(results['optimal_k'])

        # Calcul du K le plus fréquent
        from collections import Counter
        k_counts = Counter(recommendations)
        most_common_k = k_counts.most_common(1)[0][0]

        # Si pas de consensus, utiliser la médiane
        if len(set(recommendations)) == len(recommendations):  # Tous différents
            self.optimal_k = int(np.median(recommendations))
        else:
            self.optimal_k = most_common_k

        print(f"\n Résumé des recommandations:")
        print(f"   - Méthode du coude: K = {self.metrics_results['elbow']['optimal_k']}")
        print(f"   - Coefficient de silhouette: K = {self.metrics_results['silhouette']['optimal_k']}")
        print(f"   - Indice de Calinski-Harabasz: K = {self.metrics_results['calinski_harabasz']['optimal_k']}")
        print(f"   - Indice de Davies-Bouldin: K = {self.metrics_results['davies_bouldin']['optimal_k']}")
        print(f"\n Nombre optimal de clusters retenu: K = {self.optimal_k}")

        return self.optimal_k

    def fit_final_model(self):
        """
        Entraîne le modèle K-Means final avec le nombre optimal de clusters
        """
        print(f"\n🚀 Entraînement du modèle K-Means final avec K = {self.optimal_k}...")

        self.kmeans_model = KMeans(
            n_clusters=self.optimal_k,
            random_state=42,
            n_init=10,
            max_iter=300
        )

        self.cluster_labels = self.kmeans_model.fit_predict(self.scaled_data)

        # Calcul des métriques finales
        silhouette_final = silhouette_score(self.scaled_data, self.cluster_labels)
        calinski_final = calinski_harabasz_score(self.scaled_data, self.cluster_labels)
        davies_final = davies_bouldin_score(self.scaled_data, self.cluster_labels)

        print(f"✓ Modèle entraîné avec succès!")
        print(f"   - Coefficient de silhouette: {silhouette_final:.3f}")
        print(f"   - Indice de Calinski-Harabasz: {calinski_final:.2f}")
        print(f"   - Indice de Davies-Bouldin: {davies_final:.3f}")

        return self.kmeans_model

    def analyze_clusters(self):
        """
        Analyse détaillée des clusters formés
        """
        print(f"\n Analyse des clusters:")

        # Ajout des labels aux données originales
        data_with_clusters = self.data_numeric.copy()
        data_with_clusters['Cluster'] = self.cluster_labels

        # Statistiques par cluster
        cluster_stats = data_with_clusters.groupby('Cluster').agg(['mean', 'std', 'count'])

        print(f"\n📈 Distribution des clusters:")
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.cluster_labels)) * 100
            print(f"   - Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

        # Sauvegarde des résultats
        self.cluster_analysis = {
            'data_with_clusters': data_with_clusters,
            'cluster_stats': cluster_stats,
            'cluster_counts': cluster_counts
        }

        return cluster_stats

    def create_visualizations(self):
        """
        Crée les graphiques pour visualiser les résultats
        """
        print("\n Génération des graphiques...")

        # Configuration pour éviter les problèmes d'affichage
        plt.style.use('default')
        fig_size = (15, 12)

        # Création de la figure avec sous-graphiques
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle('Analyse K-Means - Détermination du nombre optimal de clusters', fontsize=16, fontweight='bold')

        # 1. Méthode du coude
        ax1 = axes[0, 0]
        elbow_data = self.metrics_results['elbow']
        ax1.plot(elbow_data['k_values'], elbow_data['inertias'], 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=elbow_data['optimal_k'], color='red', linestyle='--', alpha=0.7,
                    label=f'K optimal = {elbow_data["optimal_k"]}')
        ax1.set_xlabel('Nombre de clusters (K)')
        ax1.set_ylabel('Inertie')
        ax1.set_title('Méthode du Coude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Coefficient de silhouette
        ax2 = axes[0, 1]
        sil_data = self.metrics_results['silhouette']
        ax2.plot(sil_data['k_values'], sil_data['scores'], 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=sil_data['optimal_k'], color='red', linestyle='--', alpha=0.7,
                    label=f'K optimal = {sil_data["optimal_k"]}')
        ax2.set_xlabel('Nombre de clusters (K)')
        ax2.set_ylabel('Score de Silhouette')
        ax2.set_title('Coefficient de Silhouette')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Indice de Calinski-Harabasz
        ax3 = axes[0, 2]
        ch_data = self.metrics_results['calinski_harabasz']
        ax3.plot(ch_data['k_values'], ch_data['scores'], 'mo-', linewidth=2, markersize=8)
        ax3.axvline(x=ch_data['optimal_k'], color='red', linestyle='--', alpha=0.7,
                    label=f'K optimal = {ch_data["optimal_k"]}')
        ax3.set_xlabel('Nombre de clusters (K)')
        ax3.set_ylabel('Score Calinski-Harabasz')
        ax3.set_title('Indice de Calinski-Harabasz')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Indice de Davies-Bouldin
        ax4 = axes[1, 0]
        db_data = self.metrics_results['davies_bouldin']
        ax4.plot(db_data['k_values'], db_data['scores'], 'co-', linewidth=2, markersize=8)
        ax4.axvline(x=db_data['optimal_k'], color='red', linestyle='--', alpha=0.7,
                    label=f'K optimal = {db_data["optimal_k"]}')
        ax4.set_xlabel('Nombre de clusters (K)')
        ax4.set_ylabel('Score Davies-Bouldin')
        ax4.set_title('Indice de Davies-Bouldin')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Visualisation des clusters (PCA 2D)
        ax5 = axes[1, 1]
        if self.scaled_data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(self.scaled_data)
            scatter = ax5.scatter(data_pca[:, 0], data_pca[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.7)
            ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax5.set_title('Clusters (PCA 2D)')
            plt.colorbar(scatter, ax=ax5)
        else:
            ax5.scatter(self.scaled_data[:, 0], self.scaled_data[:, 1], c=self.cluster_labels, cmap='viridis',
                        alpha=0.7)
            ax5.set_xlabel('Feature 1')
            ax5.set_ylabel('Feature 2')
            ax5.set_title('Clusters')

        # 6. Distribution des clusters
        ax6 = axes[1, 2]
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        bars = ax6.bar(range(len(cluster_counts)), cluster_counts.values, color='skyblue', alpha=0.7)
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Nombre de points')
        ax6.set_title('Distribution des points par cluster')
        ax6.set_xticks(range(len(cluster_counts)))
        ax6.set_xticklabels([f'C{i}' for i in cluster_counts.index])

        # Ajout des valeurs sur les barres
        for bar, count in zip(bars, cluster_counts.values):
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        print("✓ Graphiques générés avec succès!")

    def get_cluster_insights(self):
        """
        Génère des insights sur les clusters pour le système de notifications

        Returns:
            dict: Insights et recommandations
        """
        print("\n💡 Génération d'insights pour le système de notifications...")

        data_with_clusters = self.cluster_analysis['data_with_clusters']
        cluster_stats = self.cluster_analysis['cluster_stats']

        insights = {
            'optimal_clusters': self.optimal_k,
            'cluster_profiles': {},
            'recommendations': []
        }

        # Profil de chaque cluster
        for cluster_id in range(self.optimal_k):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(data_with_clusters)) * 100

            # Caractéristiques moyennes du cluster
            cluster_means = cluster_data.drop('Cluster', axis=1).mean()

            insights['cluster_profiles'][cluster_id] = {
                'size': cluster_size,
                'percentage': cluster_percentage,
                'characteristics': cluster_means.to_dict()
            }

        # Recommandations pour le système de notifications
        insights['recommendations'] = [
            f"Segmentez vos utilisateurs en {self.optimal_k} groupes distincts",
            "Personnalisez les notifications selon les caractéristiques de chaque cluster",
            "Priorisez les clusters les plus importants en termes de taille",
            "Adaptez la fréquence et le contenu des notifications par segment"
        ]

        return insights

    def run_complete_analysis(self):
        """
        Lance l'analyse complète K-Means

        Returns:
            dict: Résultats complets de l'analyse
        """
        print(" Lancement de l'analyse K-Means complète...")
        print("=" * 60)

        # 1. Chargement et préprocessing
        if not self.load_and_preprocess_data():
            return None

        # 2. Détermination du K optimal
        optimal_k = self.determine_optimal_k()

        # 3. Entraînement du modèle final
        self.fit_final_model()

        # 4. Analyse des clusters
        self.analyze_clusters()

        # 5. Visualisations
        self.create_visualizations()

        # 6. Génération d'insights
        insights = self.get_cluster_insights()

        print("\n" + "=" * 60)
        print(" Analyse K-Means terminée avec succès!")

        return {
            'model': self.kmeans_model,
            'optimal_k': self.optimal_k,
            'cluster_labels': self.cluster_labels,
            'insights': insights,
            'metrics': self.metrics_results
        }