from turtle import pd
import numpy as np
import pandas as pd
from graph_matrices_completed import (
    degree_matrix,
    laplacian_pseudoinverse,
)


def common_neighbors_matrix(A: np.ndarray) -> np.ndarray:

    CN = A@A*degree_matrix(A)
    CN = np.zeros_like(A, dtype=float)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            prod = 0.0
            for k in range(n):
                prod += A
    return CN

def preferential_attachment_matrix(A: np.ndarray) -> np.ndarray:
    D_mat = degree_matrix(A, "out")
    d = np.diag(D_mat)
    PA = np.outer(d, d)
    return PA

def cosine_similarity_matrix(A: np.ndarray) -> np.ndarray:
    numerator = np.dot(A, A.T)
    
    PA = preferential_attachment_matrix(A)
    deno = np.sqrt(PA)
    
    similarity = numerator / deno
    
    return similarity


def dice_similarity_matrix(A: np.ndarray) -> np.ndarray:
    
    num = np.zeros_like(A)
    denum = np.zeros_like(A)

    d_out = np.diag(degree_matrix(A, "out"))
    d_in = np.diag(degree_matrix(A, "in"))
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            common_neighbors = 0
            for k in range(n):
                prod += A[i, k] * A[j, k]
            num[i, j] = prod
            denum[i, j] = d_out[i] + d_in[j]
    Jaccard = num/denum
    return Jaccard


def jaccard_similarity_matrix(A: np.ndarray) -> np.ndarray:
    intersection = np.dot(A, A.T)

    D_mat = degree_matrix(A, "out")
    d = np.diag(D_mat)
    
    degree_sum = d[:, None] + d[None, :]
    
    denominator = degree_sum - intersection
    
    similarity = intersection / denominator
    
    return similarity


def katz_matrix(A: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Katz = (I - alpha A)^(-1) - I
    """
    n = A.shape[0]
    I = np.eye(n)

    S_katz = (np.linalg.inv(I - alpha * A)) - I

    return S_katz




def first_passage_time_matrix(A: np.ndarray) -> np.ndarray:

    L_plus = laplacian_pseudoinverse(A)
    
    D_mat = degree_matrix(A, "out") 
    d = np.diag(D_mat)
    vol = np.sum(d)
    
    v = L_plus @ d
    
    term1 = v[:, np.newaxis]
    term2 = vol * L_plus
    term3 = v[np.newaxis, :]
    term4 = vol * np.diag(L_plus)[np.newaxis, :]
    
    M = term1 - term2 - term3 + term4
    
    np.fill_diagonal(M, 0.0)
    
    return M


def commute_time_matrix(A: np.ndarray) -> np.ndarray:
    """
    C_{ij} = vol(G) * (L⁺_{ii} + L⁺_{jj} - 2 L⁺_{ij})
    """
    L_plus = laplacian_pseudoinverse(A)
    
    vol = np.sum(A)

    lp_diag = np.diag(L_plus)
    
    term1 = lp_diag[:, np.newaxis]
    term2 = lp_diag[np.newaxis, :]
    
    C = vol * (term1 + term2 - 2 * L_plus)
    
    np.fill_diagonal(C, 0.0)
    
    return C


if __name__ == "__main__":

    nom_fichier = '.patagonia/similarity_matrix_unigram.xlsx'
    df = pd.read_excel(nom_fichier, index_col=0, engine='openpyxl')
    A = df.values
    print(A)

    print("Common neighbors :\n", common_neighbors_matrix(A))
    print("Preferential attachment index:\n", preferential_attachment_matrix(A))
    print("Cosine coefficient:\n", np.round(cosine_similarity_matrix(A), 3))
    print("Dice coefficient:\n", np.round(dice_similarity_matrix(A), 3))
    print("Jaccard coefficient:\n", np.round(jaccard_similarity_matrix(A), 3))

    # Katz
    K = katz_matrix(A, alpha=0.2)
    print("\nKatz:\n", np.round(K, 4))

    # FTP / CT 
    M = first_passage_time_matrix(A)
    N = commute_time_matrix(A)
    print("\nFTP matrix (H):\n", np.round(M, 4))
    print("\nCT matrix (C):\n", np.round(N, 4))
