import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from kneed import KneeLocator  
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

#funzione per calcolare l'sse di un clustering
class Utils_functions():
    

    def compute_SSE(X, labels):
        """
        Calcola la Somma degli Errori Quadratici (SSE) per un generico clustering.
        """
        
        #inizializzo la sse
        sse = 0.0
        
        #estraggo le etichette uniche dei cluster per calcolare la sse
        unique_labels = np.unique(labels)
        
        #per ogni etichetta calcolo la distanza tra i punti del cluster e il centroide
        for label in unique_labels:
            
            #salto l'etichetta 'rumore' (-1 in DBSCAN), se presente
            if label == -1:
                continue
            
            #prendo tutti i punti del cluster che hanno l'etichetta 'label' uguale a quella corrente
            cluster_points = X[labels == label]
            
            #se il cluster è vuoto salto il calcolo
            if cluster_points.size == 0:
                continue
            
            #calcolo il centroide del cluster come la media dei punti del cluster
            centroid = cluster_points.mean(axis=0)
            
            #distanze dei punti del cluster dal proprio centroide
            distances = pairwise_distances(cluster_points, [centroid], metric='euclidean')
            
            #aggiorno la sse sommando i quadrati delle distanze
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
    
    
    
    def choose_optimal_pca_components(X, variance_threshold=0.9):
        """
        Determina il numero minimo di componenti principali necessari
        per spiegare almeno 'variance_threshold' della varianza cumulata.
        """
        

        #effettuo una copia dei dati
        X_to_pca = X.copy()  
        
        #PCA con n_components = None estrae tutte le componenti possibili
        pca = PCA(n_components = None)
        
        #con fit a differenza di fit_transform non ritorna i dati trasformati
        #che non mi servono per calcolare la varianza cumulata
        pca.fit(X_to_pca)

        #estraggo la varianza spiegata da ogni componente
        var_ratios = pca.explained_variance_ratio_
        
        #calcolo la somma cumulata della varianza spiegata per ogni componente
        var_cum = np.cumsum(var_ratios)
        
        #calcolo il numero di componenti necessarie per spiegare almeno il 90% della varianza
        n_components = np.searchsorted(var_cum, variance_threshold) + 1
        
        print(f"Numero di componenti necessarie per spiegare almeno il {variance_threshold * 100}% della varianza: {n_components}")
        
        return n_components
    
    
    def process_data(X, variance_threshold = 0.9):
        
        """
        Preprocessa i dati in quattro modi diversi: 
        - Raw
        - Standardized
        - Normalized
        - Extracted (PCA) con un numero di componenti principali sufficienti a spiegare almeno il 90% della varianza
        """
        
        scaler = StandardScaler()

        #standardizzo i dati (media 0 e varianza 1)
        X_std = scaler.fit_transform(X)

        #normalizzo i dati (valori tra 0 e 1)
        X_norm = (X - X.min()) / (X.max() - X.min())
        
        #cerco il numero di componenti principali necessarie per spiegare almeno il 90% della varianza
        n_components = Utils_functions.choose_optimal_pca_components(X, variance_threshold)

        #applico pca con 8 componenti principali
        pca = PCA(n_components = n_components)
        X_pca = pca.fit_transform(X_std)

        #creo una lista con i tre dataset
        datasets = [X, X_std, X_norm, X_pca]
        datasets_names = ["Raw", "Standardized", "Normalized", "Extracted"]
        
        return datasets, datasets_names, n_components
    
    
    def choose_optimal_clusters_elbow_method(X, dataset_name):
        
        """
        Determina (tramite l’Elbow Method) un numero ottimale di cluster con Hierarchical 
        Clustering, confrontandolo poi con il numero reale di cluster (10)
        """
        
        #inizializza una lista vuota da riempire con i valori di sse
        sse_values = []  

        #cicla per tutti i numeri dei cluster
        for n_clusters in range(2, 16):
            hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            labels = hierarchical_clustering.fit_predict(X)

            #calcola sse
            sse = Utils_functions.compute_SSE(X, labels)
            sse_values.append(sse)

        #troviamo l'elbow point
        kneedle = KneeLocator(range(2, 16), sse_values, curve="convex", direction="decreasing")
        optimal_clusters = kneedle.knee  

        print(f"Numero ottimale di cluster (Elbow) per Dataset {dataset_name}: {optimal_clusters}")
        print(f"Numero reale di cluster: 10")
        
        if optimal_clusters == 10:
            print("Il numero reale di cluster coincide con il numero ottimale secondo la regola dell'elbow.")
        else:
            print("Il numero reale di cluster NON coincide con il numero ottimale secondo la regola dell'elbow.")

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 16), sse_values, 'bx-')
        plt.xlabel('Numero di Cluster (N)')
        plt.ylabel('SSE')
        plt.title(f'Metodo del Gomito (Elbow Method) - Dataset: {dataset_name}')
        plt.axvline(x=optimal_clusters, color='g', linestyle='--', label=f'Cluster ottimale ({optimal_clusters})')
        plt.axvline(x=10, color='r', linestyle='--', label='Numero reale cluster (10)')
        plt.legend()
        plt.grid(True)
        plt.show()


    
