from ml_supervised import RandomForest, load_data

df = load_data("data/processed/final_dataset.csv", sample_size=10000)

# Étape  pour estimer un bon k
#evaluate_kmeans_k(df)
RandomForest(df)
