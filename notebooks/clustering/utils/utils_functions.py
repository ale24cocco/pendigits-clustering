import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml

#funzione per calcolare l'sse di un clustering
class Utils_functions():
    
    
    def compute_sse(X, labels):
        """
        Calcola la somma delle distanze quadrate (SSE) 
        dei punti dai centroidi dei rispettivi cluster.
        Esclude -1 (noise) se presente.
        Restituisce un singolo float.
        """
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        sse = 0.0
        for cluster in unique_labels:
            cluster_points = X[labels == cluster]
            if len(cluster_points) == 0:
                continue
            centroid = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - centroid)**2)
        
        return float(sse)
    
    
    def load_data():
        """
        Carica il dataset pendigits.
        Se il dataset è già presente lo carica.
        Altrimenti lo scarica e lo inizializza.
        """
    
        #se il dataset è già presente lo carico
        if os.path.isfile("../../data/raw/pendigits.csv"):
            
            ds = pd.read_csv("../../data/raw/pendigits.csv")
            
            #estraggo le features e i targets
            X = ds.drop(columns = "target")
            y = ds["target"]
            
        #altrimenti lo scarico e lo inzializzo
        else:   
            
            #recupero il dataset dal modulo sklearn
            pendigits = fetch_openml(name = "pendigits", version=1, as_frame=True)
            
            #creo un dataframe con i dati e le etichette
            X = pendigits.data
            y = pendigits.target
            
            #estraggo il numero di colonne del dataframe
            n_col = X.shape[1]
            
            #rinomino le colonne del dataframe
            X.columns = [
            f"x{index}" if i % 2 == 0 else f"y{index}"
            for i in range(n_col)
            for index in [i // 2]
            ]
            
            #unisco i dati e le etichette in un unico dataframe
            df = X.copy()
            df['target'] = y
            
            #salvo il dataframe in formato csv
            df.to_csv("../../data/raw/pendigits.csv", index = False)
    
        return X, y

