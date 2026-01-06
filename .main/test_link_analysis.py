import numpy as np
import pandas as pd

# Chargement des données
nom_fichier = '.patagonia/similarity_matrix_unigram.xlsx'
df = pd.read_excel(nom_fichier, index_col=0, engine='openpyxl')
A = df.values

# --- 1. Degree : Pas de changement majeur, juste clarification ---
def degree_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    if direction == "out":
        degrees = np.sum(A, axis=1)
    else:
        degrees = np.sum(A, axis=0)
    return np.diag(degrees)

# --- 2. Shortest Path : Correction de l'initialisation (Inversion) ---
def shortest_path_with_reconstruction(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    
    # On initialise SP avec l'infini partout
    SP = np.full((n, n), np.inf)
    
    # Matrice 'Next' pour stocker le noeud suivant
    Next = np.full((n, n), -1, dtype=int)

    # --- Initialisation Correcte ---
    for i in range(n):
        for j in range(n):
            if i == j:
                SP[i, j] = 0.0
                Next[i, j] = i 
            elif A[i, j] > 0: # Si une similarité existe
                # CORRECTION CRITIQUE : Transformation en Distance
                # Plus la similarité est grande, plus la distance est petite.
                SP[i, j] = 1 - A[i, j] 
                Next[i, j] = j
            # Sinon, on laisse SP[i,j] à np.inf

    # --- Floyd-Warshall ---
    # Note : C'est long en Python pur (O(n^3)). Soyez patient si n > 100.
    for k in range(n):
        for i in range(n):
            # Optimisation : Si pas de chemin vers k, inutile de tester
            if SP[i, k] == np.inf: 
                continue
            for j in range(n):
                if SP[i, k] + SP[k, j] < SP[i, j]:
                    SP[i, j] = SP[i, k] + SP[k, j]
                    Next[i, j] = Next[i, k]

    return SP, Next

def get_path_names(start_name: str, end_name: str, df_index, Next_matrix: np.ndarray) -> list:
    try:
        u = df_index.get_loc(start_name)
        v = df_index.get_loc(end_name)
    except KeyError as e:
        return [f"Erreur : '{e.args[0]}' inconnu"]

    if Next_matrix[u, v] == -1:
        return []

    path_indices = [u]
    curr = u
    while curr != v:
        curr = Next_matrix[curr, v]
        if curr == -1: return []
        path_indices.append(curr)
        # Sécurité boucle infinie
        if len(path_indices) > len(Next_matrix): return ["Erreur : Boucle détectée"]

    return [df_index[idx] for idx in path_indices]

# --- 3. Betweenness : Correction pondérée ---
def betweenness_centrality(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    
    dist = np.full((n, n), np.inf)
    sigma = np.zeros((n, n)) # Nombre de plus courts chemins
    
    # Initialisation
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0
                sigma[i, j] = 1
            elif A[i, j] > 0:
                # CORRECTION : On utilise la distance pondérée (1 - Sim)
                # Si vous vouliez du non-pondéré (juste le nombre de sauts), mettez 1.0
                dist[i, j] = 1 - A[i, j] 
                sigma[i, j] = 1

    # Floyd-Warshall augmenté
    for k in range(n):
        for i in range(n):
            if dist[i, k] == np.inf: continue
            for j in range(n):
                if dist[k, j] == np.inf: continue
                
                new_dist = dist[i, k] + dist[k, j]
                
                if new_dist < dist[i, j]:
                    dist[i, j] = new_dist
                    sigma[i, j] = sigma[i, k] * sigma[k, j]
                elif np.isclose(new_dist, dist[i, j]): # Utilisation de isclose pour les float
                    sigma[i, j] += sigma[i, k] * sigma[k, j]

    # Calcul des scores (Dependency accumulation)
    cb = np.zeros(n)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != k and j != k and i != j:
                    if dist[i, j] != np.inf and sigma[i, j] > 0:
                        # On vérifie si k est sur un plus court chemin entre i et j
                        if np.isclose(dist[i, k] + dist[k, j], dist[i, j]):
                            cb[k] += (sigma[i, k] * sigma[k, j]) / sigma[i, j]
    
    # Normalisation
    if n > 2:
        norm = 1 / ((n - 1) * (n - 2))
        cb *= norm
    return cb

# --- MAIN ---
if __name__ == "__main__":

    threshold = 0.20 
    
    # Filtrage
    A_filtered = A.copy()
    np.fill_diagonal(A_filtered, 0)
    A_filtered[A_filtered <= threshold] = 0

    print("--- Degree Centrality ---")
    # count_nonzero donne le degré non pondéré (nombre de voisins)
    degree_centrality_out = np.count_nonzero(A_filtered, axis=1)
    print(pd.Series(degree_centrality_out, index=df.index).sort_values(ascending=False).head(5))

    print("\n--- Shortest Path Calculation ---")
    # Calcul des chemins
    SP_matrix, Next_matrix = shortest_path_with_reconstruction(A_filtered)

    p1 = "Women's Long-Sleeved Rugby Top" 
    p2 = "PowSlayer Beanie" # Assurez-vous que ce nom existe dans votre Excel

    path_list = get_path_names(p1, p2, df.index, Next_matrix)
    
    print(f"Chemin : {' -> '.join(path_list) if path_list else 'Pas de chemin'}")
    
    try:
        idx1 = df.index.get_loc(p1)
        idx2 = df.index.get_loc(p2)
        dist_val = SP_matrix[idx1, idx2]
        print(f"Distance cumulée (Minimisée) : {dist_val:.4f}")
    except:
        pass

    print("\n--- Betweenness Centrality ---")
    bc_values = betweenness_centrality(A_filtered)
    bc_series = pd.Series(bc_values, index=df.index)
    print(bc_series.sort_values(ascending=False).head(10))