import numpy as np

def gradient2D(X):
    """
    Calcule l'opérateur de gradient discret 2D appliqué à une matrice X.

    Args:
        X (np.ndarray): Matrice d'entrée de forme (M, N).

    Returns:
        tuple: Un tuple contenant deux tableaux représentant le gradient horizontal (X Dh) et le gradient vertical (Dv X) respectivement.

    Raises:
        AssertionError: Si le tableau d'entrée a plus de 2 dimensions.
    """

    # Vérification que le tableau d'entrée a exactement 2 dimensions
    assert X.ndim == 2, "Le tableau d'entrée doit être 2D."

    # Calcule les différences horizontales
    Dh = np.diff(X, axis=1)
    Dh = np.concatenate((Dh, np.zeros((X.shape[0], 1))), axis=1)

    # Calcule les différences verticales
    Dv = np.diff(X, axis=0)
    Dv = np.concatenate((Dv, np.zeros((1, X.shape[1]))), axis=0)

    return Dh, Dv
