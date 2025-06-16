# run_ml.py

from ml_models import load_data, run_kmeans, run_classification

df = load_data("data/processed/final_dataset.csv")

run_kmeans(df)
run_classification(df)
