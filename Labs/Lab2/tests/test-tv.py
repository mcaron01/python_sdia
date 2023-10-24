import numpy as np
import unittest
from src.tv import tv
from src.gradient2D import gradient2D

class TestTV(unittest.TestCase):

    def test_tv(self):
        # Test avec une matrice constante
        X_carre = np.ones((3, 3))  # Matrice carrée constante
        tv_carre = tv(X_carre)
        tv_attendue_carre = 0  # La variation totale d'une matrice constante est de 0
        self.assertEqual(tv_carre, tv_attendue_carre)

        # Test avec une matrice non-constante
        X_non_carre = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tv_non_carre = tv(X_non_carre)
        tv_attendue_non_carre = 20.64911064067352  # Calculé au préalable
        self.assertAlmostEqual(tv_non_carre, tv_attendue_non_carre, places=8)

# execution des tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, defaultTest='TestTV')
