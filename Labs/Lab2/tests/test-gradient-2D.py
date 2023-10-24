import numpy as np
import unittest
from src.gradient2D import gradient2D

class TestGradient2D(unittest.TestCase):

    # Test du format de la sortie
    def test_output_format(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # on prend une matrice 3*3
        Dh, Dv = gradient2D(X)

        # On vérifie d'abord que les Dh et Dv sont bien des tableaux NumPy
        self.assertIsInstance(Dh, np.ndarray)
        self.assertIsInstance(Dv, np.ndarray)

        # On vérifie que les dimensions sont correctes
        self.assertEqual(Dh.shape, X.shape)
        self.assertEqual(Dv.shape, X.shape)

    # Test avec une matrice constante carrée et non carrée
    def test_constant_matrix(self):
        # Test avec une matrice constante carrée
        X_square = np.ones((3, 3))  # Matrice constante carrée
        Dh_square, Dv_square = gradient2D(X_square)

        # On vérifie donc que les gradients sont tous zéros
        self.assertTrue(np.all(Dh_square == 0))
        self.assertTrue(np.all(Dv_square == 0))

        # Test avec une matrice constante non carrée
        X_non_square = np.ones((2, 3))  # Matrice constante non carrée
        Dh_non_square, Dv_non_square = gradient2D(X_non_square)

        # On vérifie que les gradients sont tous zéros
        self.assertTrue(np.all(Dh_non_square == 0))
        self.assertTrue(np.all(Dv_non_square == 0))

# execution des tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, defaultTest='TestGradient2D')
