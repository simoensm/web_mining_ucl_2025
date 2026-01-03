import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
INPUT_DATASET = ".patagonia/patagonia_dataset_clusters.xlsx" # Output of step 04
INPUT_MATRIX = ".patagonia/patagonia_tfidf_matrix.pkl"            # Output of step 04
NODES_FILE = ".patagonia/patagonia_gephi_nodes.csv"
EDGES_FILE = ".patagonia/patagonia_gephi_edges.csv"

# Threshold: Only connect products if they are more than X% similar.
# 0.15 = Loose connections (Hairball graph)
# 0.40 = Tight connections (Only very similar items)
SIMILARITY_THRESHOLD = 0.20 

def export_gephi():
    if not os.path.exists(INPUT_DATASET) or not os.path.exists(INPUT_MATRIX):
        print("Error: Input files missing. Run '04_text_mining.py' first.")
        return

    print("--- Loading Pre-Computed Data ---")
    df = pd.read_excel(INPUT_DATASET)
    
    with open(INPUT_MATRIX, 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    print(f"Loaded {len(df)} products and TF-IDF matrix.")

    # --- 1. Create NODES ---
    # In Gephi, 'Id' is the unique identifier. 'Label' is what you see.
    # We also export 'Cluster' so you can color nodes by cluster in Gephi.
    print("Generating Nodes...")
    nodes = df.reset_index()[['index', 'name', 'cluster']]
    nodes.columns = ['Id', 'Label', 'Cluster']
    
    # Adding Category if it exists for extra coloring options
    if 'category' in df.columns:
        nodes['Category'] = df['category']

    nodes.to_csv(NODES_FILE, index=False)
    print(f"Saved Nodes to {NODES_FILE}")

    # --- 2. Create EDGES ---
    print("Calculating Cosine Similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # We use the upper triangle of the matrix to avoid duplicates (A->B is same as B->A)
    # k=1 excludes the diagonal (self-loops, A->A)
    upper_tri = np.triu(cosine_sim, k=1)
    
    # Find all pairs above the threshold
    rows, cols = np.where(upper_tri > SIMILARITY_THRESHOLD)
    weights = upper_tri[rows, cols]
    
    print(f"Found {len(weights)} connections (Threshold: {SIMILARITY_THRESHOLD})")

    edges = pd.DataFrame({
        'Source': rows,   # Corresponds to 'Id' in nodes file
        'Target': cols,   # Corresponds to 'Id' in nodes file
        'Weight': weights,
        'Type': 'Undirected'
    })
    
    edges.to_csv(EDGES_FILE, index=False)
    print(f"Saved Edges to {EDGES_FILE}")
    print("--- Done ---")

if __name__ == "__main__":
    export_gephi()