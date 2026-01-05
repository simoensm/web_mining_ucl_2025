import numpy as np
from shortest_path_completed import shortest_path_matrix

def closeness_centrality(A: np.ndarray) -> np.ndarray:
    """
    Standard closeness centrality:
        C(i) = (n - 1) / Σ_j d(i, j)
    Computed using the shortest path matrix.
    """
    SP = shortest_path_matrix(A)
    n = A.shape[0]
    closeness = np.zeros(n)

    for i in range(n):
        dists = SP[i, :]
        # On ne prend en compte que les distances réelles (inférieures au seuil arbitraire de ShortestPathMatrix)
        # et strictement positives (on exclut la distance à soi-même)
        valid_dists = dists[dists < 90000] 
        
        total_dist = np.sum(valid_dists)
        
        if total_dist > 0:
            closeness[i] = (n - 1) / total_dist
        else:
            closeness[i] = 0.0
            
    return closeness

def residual_closeness_centrality(A: np.ndarray) -> np.ndarray:
    """
    Residual closeness centrality:
        RCC(k) = Σ_{i ≠ j} (1 / 2^{d(i, j)})
    Computed by removing node k from the graph, recalculating
    the shortest paths on the remaining subgraph, and summing
    the inverse powers of the distances.
    """
    n = A.shape[0]
    rcc = np.zeros(n)
    
    for k in range(n):
        # 1. Créer le sous-graphe sans le noeud k
        # np.delete supprime la ligne k, puis la colonne k
        A_sub = np.delete(np.delete(A, k, axis=0), k, axis=1)
        
        # 2. Recalculer les plus courts chemins sur ce sous-graphe
        SP_sub = shortest_path_matrix(A_sub)
        
        # 3. Calculer le score
        score = 0.0
        n_sub = SP_sub.shape[0]
        
        for i in range(n_sub):
            for j in range(n_sub):
                if i != j:
                    dist = SP_sub[i, j]
                    # On ignore les distances "infinies" (100000 dans ShortestPathMatrix)
                    if dist < 90000:
                        score += 1.0 / (2.0 ** dist)
                        
        rcc[k] = score
        
    return rcc


def eccentricity_centrality(A: np.ndarray) -> np.ndarray:
    """
    Eccentricity centrality of each node:
        ecc(i) = 1 / max_j d(i, j)
    The inverse is used so that larger distances correspond
    to smaller eccentricity values. Unreachable distances
    (set to 100000) are ignored in the logic of shortest_path_matrix.
    """
    SP = shortest_path_matrix(A)
    n = A.shape[0]
    ecc = np.zeros(n)
    
    for i in range(n):
        row_dists = SP[i, :]
        
        # On cherche la distance max.
        # Attention : si le graphe n'est pas connexe, ShortestPathMatrix met 100000.
        # Si un noeud ne peut pas atteindre les autres, son excentricité (distance max) est techniquement l'infini.
        # Ici, 1/100000 donne un score très proche de 0, ce qui est logique.
        
        # Cependant, pour un graphe connexe, on filtre souvent la diagonale (0).
        # Mais le max d'une ligne contenant des nombres positifs sera > 0 de toute façon.
        
        max_dist = np.max(row_dists)
        
        if max_dist > 0:
            ecc[i] = 1.0 / max_dist
        else:
            # Cas théorique d'un nœud isolé avec lui-même seulement (dist 0)
            ecc[i] = 0.0 
            
    return ecc


def graph_radius(A: np.ndarray) -> float:
    """
    Graph radius:
        radius = min_i eccentricity(i)
    The radius is computed as the minimum of the inverted
    eccentricity values.
    """
    ecc_values = eccentricity_centrality(A)
    
    # On filtre les zéros si nécessaire, mais la consigne est simple : min_i
    if len(ecc_values) > 0:
        return np.min(ecc_values)
    return 0.0

if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    ], int)
    
    A_RC = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]],int)

    print("Closeness:\n", np.round(closeness_centrality(A), 4))
    print("Residual closeness:\n", np.round(residual_closeness_centrality(A_RC), 4))
    print("Eccentricity:\n", eccentricity_centrality(A))
    print("Radius:\n", graph_radius(A))
