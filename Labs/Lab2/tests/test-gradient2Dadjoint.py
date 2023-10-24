import numpy as np
import unittest
from src.gradient2DAdjoint import gradient2D_adjoint

class TestGradient2DAdjoint(unittest.TestCase):

    # On test l'assertion error
    def test_gradient2D_adjoint_different_dimensions(self):
        # Créer des gradients horizontaux et verticaux de test avec des dimensions différentes
        Y_h = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y_v = np.array([[9, 8, 7], [6, 5, 4]])

        # Appeler la fonction gradient2D_adjoint doit déclencher une AssertionError
        with self.assertRaises(AssertionError):
            gradient2D_adjoint(Y_h, Y_v)

     # Test avec des gradients horizontaux et verticaux de même forme
    def test_same_shape(self):
        Y_h2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y_v2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

        result = gradient2D_adjoint(Y_h2, Y_v2)

        expected_result = np.array([[-10, -9, -4], [-1, 2, 9], [-4, 1, 10]])

        np.testing.assert_array_equal(result, expected_result)

# execution des tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, defaultTest='TestGradient2DAdjoint')
