import numpy as np
from graph_matrices_completed import transition_matrix


def pagerank_eigen(A: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """
    Compute PageRank scores using the eigenvector method.

    Google matrix:
        G = α * Pᵀ + (1 - α) * E
    where:
        P is the transition probability matrix (row-stochastic),
        E is the uniform teleportation matrix (1/n for all entries).
    The dominant eigenvector of G gives the PageRank scores.
    """
    n = A.shape[0]
    
    # Matrice de transition P
    P = transition_matrix(A)
    
    # Matrice de téléportation E (1/n partout)
    E = np.ones((n, n)) / n
    
    # Matrice Google (Attention: P est transposée car on cherche le vecteur propre 
    # d'une matrice stochastique par colonne pour l'équation G * v = v)
    G = alpha * P.T + (1 - alpha) * E
    
    # Calcul vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eig(G)
    
    # Trouver l'indice de la valeur propre 1 (ou la plus proche/grande)
    idx = np.argmax(np.abs(eigenvalues))
    
    # Vecteur propre dominant
    pr = np.abs(np.real(eigenvectors[:, idx]))
    
    # Normalisation L1
    return pr / np.sum(pr)



def pagerank_power_iteration(A: np.ndarray, alpha: float = 0.85, max_iter: int = 100) -> np.ndarray:
    
    n = A.shape[0]
    P = transition_matrix(A)
    
    # Initialisation uniforme
    pr = np.ones(n) / n
    
    # Vecteur de téléportation u
    u = np.ones(n) / n
    
    for _ in range(max_iter):
        pr_next = alpha * np.dot(P.T, pr) + (1 - alpha) * u
        
        # Vérification de convergence
        if np.linalg.norm(pr_next - pr, 1) < 1e-6:
            return pr_next
        
        pr = pr_next
        
    return pr


if __name__ == "__main__":
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ], dtype=int)

    print("PageRank (eigenvector method):")
    pr_eig = pagerank_eigen(A, alpha=0.827)
    print(np.round(pr_eig, 4))

    print("\nPageRank (power iteration):")
    pr_iter = pagerank_power_iteration(A, alpha=0.827, max_iter=100)
    print(np.round(pr_iter, 4))
