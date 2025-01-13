# Pendigits Clustering

Questo progetto si focalizza sull’analisi e il clustering del dataset Pendigits, un insieme di cifre scritte a mano e rappresentate da coordinate spaziali (x, y). L’obiettivo principale è individuare strutture latenti nei dati, verificando la capacità di diversi algoritmi di clustering (K-Means, Hierarchical Agglomerative Clustering, DBSCAN) nel separare correttamente le cifre.
Dopo una fase di preprocessing, che include la standardizzazione e la riduzione della dimensionalità (PCA), vengono valutate le performance di ciascun algoritmo con metriche supervisionate (ARI, NMI, ecc.) e non supervisionate (SSE, MSE). I risultati mostrano come la scelta del preprocessing e dei parametri di clustering influenzi in modo significativo la definizione dei cluster, evidenziando anche quali cifre risultino più difficili da distinguere. L’analisi conferma l’importanza di un’adeguata normalizzazione e di un’ottimizzazione mirata dei parametri, conferendo solidi spunti sulle tecniche più efficaci per riconoscere o raggruppare dati scritti a mano.

Di seguito sono riportate le istruzioni per configurare l'ambiente di sviluppo, installare le dipendenze necessarie e avviare il progetto.

## Requisiti

-   Python 3.8 o superiore
-   `conda` (opzionale, per creare un ambiente Conda)

## Configurazione dell'Ambiente

### Utilizzando Conda

1. Assicurati di avere Anaconda o Miniconda installato.
2. Esegui il seguente comando per creare un ambiente Conda e installare le dipendenze:
    ```bash
    conda env create -f environment.yml
    ```
3. Attiva l'ambiente:
    ```bash
    conda activate pendigits-clustering
    ```

### Utilizzando venv

1. Crea un ambiente virtuale nella directory del progetto:
    ```bash
    python -m venv venv
    ```
2. Attiva l'ambiente:
    - Su Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    - Su Windows:
        ```bash
        venv\Scripts\activate
        ```
3. Installa i requisiti usando `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Struttura della Cartella

-   `venv/`: Directory contenente l'ambiente virtuale (se creato con venv).
-   `.gitignore`: File per ignorare elementi indesiderati nel repository.
-   `environment.yml`: File per creare un ambiente Conda con le dipendenze necessarie.
-   `Progetto 2.5.pdf`: Documentazione relativa al progetto.
-   `requirements.txt`: File che elenca le dipendenze necessarie per `pip`.

## Esecuzione del Progetto

1. Assicurati di aver attivato l'ambiente (`conda activate pendigits-clustering` o `source venv/bin/activate`).
2. Avvia lo script principale del progetto (specificare il nome del file se necessario):
    ```bash
    python main.py
    ```

## Note Utili

-   Se riscontri conflitti o errori durante l'installazione, verifica di avere la versione corretta di Python e delle librerie richieste.
-   Se utilizzi Conda, `environment.yml` garantisce una configurazione più precisa delle dipendenze rispetto a `requirements.txt`.

## Risorse

-   [Dataset Pendigits](https://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits): Per ulteriori dettagli sul dataset.

---

Per qualsiasi problema, contattare il responsabile del progetto o fare riferimento alla documentazione inclusa nel file `Progetto 2.5.pdf`.
