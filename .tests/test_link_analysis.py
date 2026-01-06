import numpy as np
import pandas as pd

Treshold = 0.20 #à remplacer avec text_mining_main car sinon resultat non identiques sur gephi et ici
# Chargement des données
nom_fichier = '.patagonia/similarity_matrix_unigram.xlsx'
df = pd.read_excel(nom_fichier, index_col=0, engine='openpyxl')
S = df.values

A = (S > Treshold).astype(int)

np.fill_diagonal(A, 0)

print(S)
print(A)

def degree_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    if direction == "out":
        degrees = np.sum(A, axis=1)
    else:
        degrees = np.sum(A, axis=0)

    return np.diag(degrees)

def shortest_path_with_reconstruction(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    SP = A.copy().astype(float)
    
    # Matrice 'Next' pour stocker le noeud suivant dans le chemin
    # Initialisation avec -1 (pas de chemin)
    Next = np.full((n, n), -1, dtype=int)

    # --- Initialisation ---
    for i in range(n):
        for j in range(n):
            if i == j:
                SP[i, j] = 0.0
                Next[i, j] = i # Sur soi-même
            elif SP[i, j] != 0:
                # S'il y a un lien direct, le successeur de i vers j est j
                # (On suppose ici que A contient des distances/coûts)
                Next[i, j] = j 
            else:
                # Pas de lien direct, on met l'infini
                SP[i, j] = 100000.0 

    # --- Floyd-Warshall avec reconstruction de chemin ---
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Si passer par k est plus court que le chemin actuel i->j
                if SP[i, k] + SP[k, j] < SP[i, j]:
                    SP[i, j] = SP[i, k] + SP[k, j]
                    # Le prochain noeud pour aller de i à j devient le même que pour aller de i à k
                    Next[i, j] = Next[i, k]

    return SP, Next

def get_path_names(start_name: str, end_name: str, df_index, Next_matrix: np.ndarray) -> list:
    try:
        u = df_index.get_loc(start_name)
        v = df_index.get_loc(end_name)
    except KeyError as e:
        return [f"Erreur : Le produit '{e.args[0]}' n'existe pas."]

    # S'il n'y a pas de chemin (le successeur est -1)
    if Next_matrix[u, v] == -1:
        return [] # Pas de chemin

    path_indices = [u]
    curr = u
    
    # On navigue de successeur en successeur jusqu'à arriver à v
    while curr != v:
        curr = Next_matrix[curr, v]
        # Sécurité : si on tombe sur -1 en cours de route
        if curr == -1:
            return [] 
        path_indices.append(curr)

    # Conversion des indices en noms de produits
    path_names = [df_index[idx] for idx in path_indices]
    return path_names

def betweenness_centrality(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    
    dist = np.full((n, n), np.inf)
    sigma = np.zeros((n, n))
    
    # Initialisation
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0
                sigma[i, j] = 1
            elif A[i, j] != 0:
                dist[i, j] = 1.0/A[i, j]
                sigma[i, j] = 1

    # Floyd-Warshall augmenté
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    sigma[i, j] = sigma[i, k] * sigma[k, j]
                elif dist[i, k] + dist[k, j] == dist[i, j] and dist[i, j] != np.inf:
                    sigma[i, j] += sigma[i, k] * sigma[k, j]

    # Calcul des scores
    cb = np.zeros(n)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != k and j != k and i != j:
                    if dist[i, j] != np.inf and sigma[i, j] > 0:
                        if dist[i, k] + dist[k, j] == dist[i, j]:
                            cb[k] += (sigma[i, k] * sigma[k, j]) / sigma[i, j]
    
    
    cb /= 2.0

    return cb

if __name__ == "__main__":

    

    #A_filtered = A.copy()

    #np.fill_diagonal(A_filtered, 0)

    #A_filtered[A_filtered <= threshold] = 0

    print("Degree Centrality :")
    D_out = degree_matrix(A, "out") 
    degree_centrality_out = np.count_nonzero(A, axis=1)
    moyenne = np.mean(degree_centrality_out)
    
    degree_series = pd.Series(degree_centrality_out, index=df.index)
    print("Top 10 Degree Centrality :")
    print(degree_series.sort_values(ascending=False).head(10))

    print(f"Moyenne : {moyenne}")



    print("\nShortest Path :")
    # On récupère à la fois les distances (SP) et les successeurs (Next)
    SP_matrix, Next_matrix = shortest_path_with_reconstruction(A)

    # Définition des produits
    p1 = "Women's Long-Sleeved Rugby Top" 
    p2 = "PowSlayer Beanie"
 
    # Récupération de la distance numérique
    try:
        idx1 = df.index.get_loc(p1)
        idx2 = df.index.get_loc(p2)
        dist_val = SP_matrix[idx1, idx2]
    except KeyError:
        dist_val = float('inf')

    # Récupération du chemin complet (liste de noms)
    path_list = get_path_names(p1, p2, df.index, Next_matrix)

    print(f"\nChemin de : '{p1}'")
    print(f"Vers      : '{p2}'")
    
    if dist_val >= 100000:
        print("Pas de chemin possible ")
    else:
        print(f"Distance  : {dist_val}")
        print("Etapes    : " + " -> ".join(path_list))



    print("\nBetweenness Centrality :")
    bc_values = betweenness_centrality(A)
    bc_series = pd.Series(bc_values, index=df.index)
    print(bc_series.sort_values(ascending=False).head(10))