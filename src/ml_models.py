import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def evaluate_kmeans_k(df: pd.DataFrame, max_k: int = 6, sample_size: int = 3000):
    df_clust = df.dropna(subset=["cvss_score", "epss_score", "epss_percentile", "days_open"])

    if len(df_clust) > sample_size:
        df_clust = df_clust.sample(n=sample_size, random_state=42)

    X = df_clust[["cvss_score", "epss_score", "epss_percentile", "days_open"]]
    X_scaled = StandardScaler().fit_transform(X)

    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)

        score = silhouette_score(X_scaled, labels)
        silhouettes.append(score)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertias, marker='o')
    plt.title("Méthode du coude (inertie)")
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Inertie")

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouettes, marker='o', color='orange')
    plt.title("Score silhouette moyen")
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Silhouette")

    plt.tight_layout()
    plt.show()

def run_kmeans(df: pd.DataFrame, n_clusters: int = 3):
    df_clust = df.dropna(subset=["cvss_score", "epss_score", "epss_percentile", "days_open"])
    X = df_clust[["cvss_score", "epss_score", "epss_percentile", "days_open"]]
    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title("Clusters de vulnérabilités (KMeans)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def run_classification(df: pd.DataFrame):
    df = df.dropna(subset=["cvss_sev", "cvss_score", "epss_score", "epss_percentile", "days_open"])
    X = df[["cvss_score", "epss_score", "epss_percentile", "days_open"]]
    y = df["cvss_sev"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.sort_values().plot(kind='barh')
    plt.title("Importance des variables (RandomForest)")
    plt.show()
