import numpy as np
from src.gradient2D import gradient2D

def tv(X):
    """
    Calcule la variation totale isotropique discrète (TV) d'une matrice X.

    Args:
        X (np.ndarray): Matrice d'entrée de forme (M, N).

    Returns:
        float: Variation totale de la matrice d'entrée.

    Raises:
        AssertionError: Si le tableau d'entrée a plus de 2 dimensions.
    """
    # Calcul du gradient discret 2D
    Dh, Dv = gradient2D(X)

    # Calcul de la TV
    total_variation = np.sum(np.sqrt(Dh**2 + Dv**2))

    return total_variation
