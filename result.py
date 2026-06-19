import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

chemin_z_score = r"C:\Users\felic\Documents\ifri_mini_ml_lib\ifri_mini_ml_lib\anomalies_detection\z_score.py"

if os.path.exists(chemin_z_score):
    print(f"Fichier trouvé: {chemin_z_score}")
else:
    print(f"Fichier NON trouvé: {chemin_z_score}")
    exit(1)

spec = importlib.util.spec_from_file_location("z_score", chemin_z_score)
z_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(z_score)

modified_zscore_detection = z_score.modified_zscore_detection


df = pd.read_csv('creditcard.csv')
data = df['Amount'].values


THRESHOLD = 3.5
anomalies_modified, mzscores = modified_zscore_detection(
    data, 
    threshold=THRESHOLD, 
    return_zscore=True
)

n_anomalies = sum(anomalies_modified)
print(f"Anomalies détectées : {n_anomalies}")
print(f"Taux d'anomlies détectées : {sum(anomalies_modified)/len(anomalies_modified)*100:.2f}%")


