import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


def load_data(csv_path: str,  sample_size: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df

def RandomForest(df: pd.DataFrame) -> pd.DataFrame:

    # Supprimer les lignes avec des valeurs manquantes dans les colonnes nécessaires
    X = df.dropna(subset=["cvss_score", "n_cve_refs","days_open", "epss_score"]).copy()

    # Garder uniquement les colonnes nécessaires
    X = X[["cvss_score", "n_cve_refs", "days_open", "epss_score"]]

    # Définir la cible (y) en classes discrètes
    def map_cvss_to_class(score):
        if score < 4.0:
            return "LOW"
        elif 4.0 <= score < 7.0:
            return "MEDIUM"
        else:
            return "HIGH"

    y = X["cvss_score"].apply(map_cvss_to_class)
    X = X.drop(columns=["cvss_score"])

    # Convertir les colonnes en types appropriés
    X["epss_score"] = pd.to_numeric(X["epss_score"], errors='coerce')
    X["n_cve_refs"] = pd.to_numeric(X["n_cve_refs"], errors='coerce')
    X["days_open"] = pd.to_numeric(X["days_open"], errors='coerce')

    # Convertir les colonnes catégoriques en numériques
    X = pd.get_dummies(X, drop_first=True)

    # Remplacer les NaN restants
    X = X.fillna(0)

    # Visualiser la distribution des classes
    print(Counter(y))

    # Appliquer SMOTE pour équilibrer les classes
    X_b, y_b = SMOTE().fit_resample(X, y)

    # Visualiser la distribution des classes
    print(Counter(y_b))

    # Diviser les données en ensembles d'entraînement et de test
    X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=0.2, random_state=42)

    # Modèle RandomForest
    rf_b = RandomForestClassifier(random_state=42)
    rf_b.fit(X_tr, y_tr)
    y_pred_b = rf_b.predict(X_te)

    # Évaluer les performances
    acc = accuracy_score(y_te, y_pred_b) * 100
    pre = precision_score(y_te, y_pred_b, average='micro')
    rec = recall_score(y_te, y_pred_b, average='micro')
    f1 = f1_score(y_te, y_pred_b, average='micro')

    print("KNN - Accuracy: {:.3f}.".format(acc))
    print("KNN - Precision: {:.3f}.".format(pre))
    print("KNN - Recall: {:.3f}.".format(rec))
    print("KNN - F1 Score: {:.3f}.".format(f1))

    print("\nClassification Report")
    print(classification_report(y_te, y_pred_b))

    # Matrice de confusion
    cm = confusion_matrix(y_te, y_pred_b, labels=["LOW", "MEDIUM", "HIGH"])
    print(cm)

    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, linewidths=.5, square=True, cmap='PuBuGn', xticklabels=["LOW", "MEDIUM", "HIGH"], yticklabels=["LOW", "MEDIUM", "HIGH"])
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title("Confusion Matrix", size=12)
    plt.show()

    feature_importances = pd.Series(rf_b.feature_importances_, index=X.columns)
    top_20_features = feature_importances.sort_values(ascending=False).head(20)
    top_20_features.plot(kind='barh')
    plt.title("Top 20 Importance des variables (RandomForest)")
    plt.show()

    # 
    # Sélectionner 30 bulletins sans score CVSS
    df_no_cvss = df[df["cvss_score"].isna()].sample(n=30, random_state=42)

    # Préparer les données pour la prédiction
    X_no_cvss = df_no_cvss[["n_cve_refs", "days_open", "epss_score"]].copy()
    X_no_cvss = pd.get_dummies(X_no_cvss, drop_first=True).fillna(0)

    # Prédire les scores CVSS avec le modèle KNN (RandomForest)
    predicted_classes = rf_b.predict(X_no_cvss)

    # Ajouter les scores prédits à une copie du DataFrame original
    df_no_cvss = df_no_cvss.copy()
    df_no_cvss["predicted_cvss_sev"] = predicted_classes

    # Afficher les résultats
    print("Bulletins sans score CVSS avec scores prédits :")
    print(df_no_cvss[["id_anssi", "n_cve_refs", "days_open", "epss_score", "predicted_cvss_sev"]])

    
    # Créer un histogramme initial basé sur les données existantes
    plt.figure(figsize=(8, 6))
    df['cvss_sev'] = df['cvss_score'].apply(map_cvss_to_class)
    sns.countplot(data=df, x='cvss_sev', order=["LOW", "MEDIUM", "HIGH"], palette="viridis")
    plt.title("Distribution des bulletins par cvss_sev (avant prédiction)")
    plt.xlabel("cvss_sev")
    plt.ylabel("Nombre de bulletins")
    plt.show()

    # Ajouter les scores prédits au DataFrame original
    df.loc[df_no_cvss.index, "cvss_sev"] = predicted_classes

    # Créer un nouvel histogramme avec les données mises à jour
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='cvss_sev', order=["LOW", "MEDIUM", "HIGH"], palette="viridis")
    plt.title("Distribution des bulletins par cvss_sev (après prédiction)")
    plt.xlabel("cvss_sev")
    plt.ylabel("Nombre de bulletins")
    plt.show()
    

    return df
