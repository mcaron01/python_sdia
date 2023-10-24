import numpy as np


def markov(rho, A, nmax, rng):
    """
    Simule une chaîne de Markov discrète homogène.

    Args:
        rho (numpy.ndarray): Loi de l'état initial
        A (numpy.ndarray): Matrice de transition (de taille NxN).
        nmax (int): Nombre d'étapes de temps.
        rng (numpy.random.Generator): Générateur de nombres aléatoires.

    Returns:
        numpy.ndarray: Trajectoire de la chaîne.
    """

    # On vérifie que rho est un vecteur de probabilités valide
    assert np.isclose(np.sum(rho), 1), "Le vecteur rho doit sommer à 1"
    assert np.all(rho >= 0), "Toutes les valeurs de rho doivent être non-négatives"

    # On vérifie ensuite que A est une matrice de transition valide
    assert np.all(A >= 0), "Tous les éléments de A doivent être non-négatifs"
    assert np.allclose(np.sum(A, axis=1), 1), "Chaque ligne de A doit sommer à 1"

    N = len(rho)
    X = np.zeros(
        nmax, dtype=int
    )  # Initialisation du tableau pour stocker la trajectoire

    # Tirer X_0 selon la loi rho
    X[0] = rng.choice(np.arange(1, N + 1), p=rho)

    # Simuler la trajectoire de la chaîne
    for q in range(nmax - 1):
        X[q + 1] = rng.choice(np.arange(1, N + 1), p=A[X[q] - 1])

    return X
