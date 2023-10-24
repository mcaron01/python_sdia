import numpy as np

def gradient2D_adjoint(Y_h, Y_v):
    """
    Calcule l'adjoint de l'opérateur de gradient discret 2D appliqué à Y.

    Args:
        Y_h (np.ndarray): Gradient horizontal de forme (M, N).
        Y_v (np.ndarray): Gradient vertical de forme (M, N).

    Returns:
        np.ndarray: Résultat de l'opération adjointe de l'opérateur de gradient discret 2D appliqué à Y.

    Raises:
        AssertionError: Si les tableaux d'entrée n'ont pas les mêmes dimensions.
    """
    # Vérification que les tableaux d'entrée ont les mêmes dimensions
    assert Y_h.shape == Y_v.shape, "Les tableaux d'entrée doivent avoir les mêmes dimensions."

    # Calcul de l'opération adjointe

    # Calcul des différences horizontales négatives
    D_adj_h = -np.diff(Y_h, axis=1)
    D_adj_h = np.concatenate((-Y_h[:, :1], D_adj_h[:, :-1], Y_h[:, -1:]), axis=1)

    # Calcul des différences verticales négatives
    D_adj_v = -np.diff(Y_v, axis=0)
    D_adj_v = np.concatenate((-Y_v[:1, :].T, D_adj_v[:-1, :].T, Y_v[-1:, :].T), axis=1)

    D_adj_v = D_adj_v.T

    return D_adj_h + D_adj_v
