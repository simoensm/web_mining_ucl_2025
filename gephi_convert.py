import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DATASET = ".patagonia/patagonia_dataset_clusters.xlsx"
INPUT_MATRIX = ".patagonia/patagonia_tfidf_matrix.pkl"  
NODES_FILE = ".patagonia/patagonia_gephi_nodes.csv"
EDGES_FILE = ".patagonia/patagonia_gephi_edges.csv"

SIMILARITY_THRESHOLD = 0.20 # Choix du seuil de similarité pour créer une arête

def export_gephi():
    if not os.path.exists(INPUT_DATASET) or not os.path.exists(INPUT_MATRIX):
        print("Error: Input files missing. Run '04_text_mining.py' first.")
        return

    print("--- Loading Pre-Computed Data ---")
    df = pd.read_excel(INPUT_DATASET)
    
    with open(INPUT_MATRIX, 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    print(f"Loaded {len(df)} products and TF-IDF matrix.")

    # Création des "nodes" pour Gephi
    print("Generating Nodes...")
    nodes = df.reset_index()[['index', 'name', 'cluster']]
    nodes.columns = ['Id', 'Label', 'Cluster']
    
    # Ajout des catégories pour vérifier clusters intra-sites
    if 'category' in df.columns:
        nodes['Category'] = df['category']

    nodes.to_csv(NODES_FILE, index=False)
    print(f"Saved Nodes to {NODES_FILE}")

    # Création des "edges" en utilisant la similarité cosinus
    print("Calculating Cosine Similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Evite les doublons en ne prenant que la moitié supérieure de la matrice (car A>B == B>A)
    upper_tri = np.triu(cosine_sim, k=1)
    
    # Trouve les paires avec une similarité au-dessus du seuil minimum
    rows, cols = np.where(upper_tri > SIMILARITY_THRESHOLD)
    weights = upper_tri[rows, cols]
    
    print(f"Found {len(weights)} connections (Threshold: {SIMILARITY_THRESHOLD})")

    edges = pd.DataFrame({
        'Source': rows,
        'Target': cols,
        'Weight': weights,
        'Type': 'Undirected'
    })
    
    edges.to_csv(EDGES_FILE, index=False)
    print(f"Saved Edges to {EDGES_FILE}")
    print("--- Done ---")

if __name__ == "__main__":
    export_gephi()