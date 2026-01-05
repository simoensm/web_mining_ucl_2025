import numpy as np


def hits(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute HITS scores (Authorities and Hubs) using the eigenvector method.

    Authority scores are the principal eigenvector of AᵀA.
    Hub scores are the principal eigenvector of AAᵀ.

    Both vectors are normalized to sum to 1.
    """
    M_auth = np.dot(A.T, A)
    # Hub matrix: A * A^T
    M_hub = np.dot(A, A.T)

    # Calcul des vecteurs propres
    eig_val_auth, eig_vec_auth = np.linalg.eig(M_auth)
    eig_val_hub, eig_vec_hub = np.linalg.eig(M_hub)

    # Récupérer l'indice de la valeur propre dominante (la plus grande en valeur absolue)
    idx_auth = np.argmax(np.abs(eig_val_auth))
    idx_hub = np.argmax(np.abs(eig_val_hub))

    # Récupérer les vecteurs propres correspondants
    # On prend la partie réelle et la valeur absolue (pour éviter les signes négatifs arbitraires)
    authority_scores = np.abs(np.real(eig_vec_auth[:, idx_auth]))
    hub_scores = np.abs(np.real(eig_vec_hub[:, idx_hub]))

    # Normalisation (somme = 1)
    if np.sum(authority_scores) > 0:
        authority_scores /= np.sum(authority_scores)
    
    if np.sum(hub_scores) > 0:
        hub_scores /= np.sum(hub_scores)

    return authority_scores, hub_scores


if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    print("HITS (eigenvector method):")
    authority_eig, hub_eig = hits(A)
    print("Authorities:", np.round(authority_eig, 4))
    print("Hubs:", np.round(hub_eig, 4))