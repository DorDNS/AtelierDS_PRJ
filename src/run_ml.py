from ml_models import load_data, run_kmeans, run_classification, evaluate_kmeans_k

df = load_data("data/processed/final_dataset.csv")

# Étape  pour estimer un bon k
#evaluate_kmeans_k(df)
run_kmeans(df, n_clusters=5)
run_classification(df)
