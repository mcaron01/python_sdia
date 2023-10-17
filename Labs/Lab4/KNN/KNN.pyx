cimport numpy as np
import numpy as np
cimport cython

cpdef k_nearest_neighbors(double[:, :] x_train,
                        double[:] class_train,
                        double[:, :] x_test,
                        int k):
    cdef Py_ssize_t num_test_samples = x_test.shape[0]
    cdef Py_ssize_t num_train_samples = x_train.shape[0]
    cdef long[:] predicted_labels = np.zeros(num_test_samples, dtype=long)
    cdef long[:] k_nearest_indices
    cdef long[:] k_nearest_labels = np.zeros(k, dtype=long)
    cdef long[:] counts
    cdef double[:] distances = np.zeros(num_train_samples, dtype=float)

    for i in range(num_test_samples):
        for q in range(num_train_samples):
            distances[q] = 0.0  # Réinitialise la distance
            for j in range(x_train.shape[1]):
                distances[q] += (x_test[i,j] - x_train[q,j]) ** 2
            distances[q] = distances[q] ** 0.5
        k_nearest_indices = np.argsort(distances)[:k].astype(long)
        for l in range (k):
            k_nearest_labels[l] = long(class_train[k_nearest_indices[l]])
            
        predicted_labels[i] = np.bincount(k_nearest_labels).astype(long).argmax()

    return predicted_labels
