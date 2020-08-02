import numpy
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,
                           n_features=20,
                           n_informative=10)

X = X.astype(numpy.float32)
y = y.astype(numpy.int64)