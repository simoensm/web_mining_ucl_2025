from turtle import pd
import numpy as np
import pandas as pd
import numpy as np

def degree_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    """
    Compute the degree matrix (either in-degree or out-degree).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n x n)
    direction : str, optional
        'out' for out-degree (default)
        'in'  for in-degree

    Returns
    -------
    D : np.ndarray
        Diagonal degree matrix
    """

    if direction == "out":
        degrees = np.sum(A, axis=1)
    else:
        degrees = np.sum(A, axis=0)

    return np.diag(degrees)



def transition_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:

    """
    Compute the transition probability matrix P.

    P[i, j] = A[i, j] / out-degree(i)
    Rows with zero degree remain zero (isolated nodes).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n x n)
    direction : str, optional
        'out' for transitions based on outgoing links (default)
    """

    D = degree_matrix(A, direction)
    deg = np.diag(D)
    P = np.zeros_like(A) 
    
    for i, d in enumerate(deg):
        if d > 0:
            P[i, :] = A[i, :] / d
    
    P = np.linalg.inv(D) @ A

    return P



def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the (combinatorial) Laplacian matrix: L = D - A

    Valid mainly for undirected graphs.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n x n)
    """
    D = degree_matrix(A, "out")
    L = D - A
    return L


def laplacian_pseudoinverse(A: np.ndarray) -> np.ndarray:
    """
    Compute the Moore–Penrose pseudo-inverse of the Laplacian matrix.

        L⁺ = (L - eeᵀ/n)^(-1) + eeᵀ/n

    This formula assumes the graph is connected.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n x n)
    """
    L = laplacian_matrix(A)
    n = A.shape[0]
    e = np.ones((n,1))
    E = (e @ e.T) / n
    M = L - E
    M_inv = np.linalg.inv(M)
    L_plus = M_inv + E
    return L_plus


if __name__ == "__main__":
    # Adjacency matrix of an example undirected graph
    # A = np.array([
    #     [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    #     [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    #     [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    # ], int)
    
    nom_fichier = '.patagonia/similarity_matrix_unigram.xlsx'
    df = pd.read_excel(nom_fichier, index_col=0, engine='openpyxl')

    A = df.values

    print("Similarity matrix A =\n", A)

    # Degree matrices
    D_out = degree_matrix(A, "out")
    D_in  = degree_matrix(A, "in")
    print("\nDegree matrix (out) D_out =\n", D_out)
    print("\nDegree matrix (in)  D_in =\n", D_in)

    # Transition matrix
    P = transition_matrix(A)
    print("\nTransition probability matrix P =\n", np.round(P, 2))

    # Laplacian
    L = laplacian_matrix(A)
    print("\nLaplacian matrix L =\n", L)

    # Laplacian pseudo-inverse
    L_plus = laplacian_pseudoinverse(A)
    print("\nLaplacian pseudo-inverse L⁺ =\n", np.round(L_plus, 4))
