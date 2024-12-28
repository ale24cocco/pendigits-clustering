import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

#funzione per calcolare l'sse di un clustering
class Utils_functions():
    

    def compute_SSE(X, labels):
        """
        Calcola la Somma degli Errori Quadratici (SSE) per un generico clustering.
        
        Parametri
        ----------
        X : array-like, shape (n_samples, n_features)
            Dati
        labels : array-like, shape (n_samples,)
            Etichette dei cluster assegnate ad ogni campione
        
        Ritorna
        -------
        sse : float
            Somma degli errori quadratici
        """
        sse = 0.0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Salta l'etichetta 'rumore' (-1 in DBSCAN), se presente
            if label == -1:
                continue
            
            cluster_points = X[labels == label]
            if cluster_points.size == 0:
                continue
            centroid = cluster_points.mean(axis=0)
            
            # Distanze dei punti del cluster dal proprio centroide
            distances = pairwise_distances(cluster_points, [centroid], metric='euclidean')
            sse += np.sum(distances ** 2)
        
        return sse
    
    
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
    
    
    def process_data(X):
        
        scaler = StandardScaler()

        #standardizzo i dati (media 0 e varianza 1)
        X_std = scaler.fit_transform(X)

        #normalizzo i dati (valori tra 0 e 1)
        X_norm = (X - X.min()) / (X.max() - X.min())

        #applico pca con 8 componenti principali
        pca = PCA(n_components=8)
        X_pca = pca.fit_transform(X_std)

        #creo una lista con i tre dataset
        datasets = [X, X_std, X_norm, X_pca]
        datasets_names = ["Raw", "Standardized", "Normalized", "Extracted"]
        
        return datasets, datasets_names

