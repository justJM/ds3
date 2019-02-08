# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

mnist = fetch_openml('mnist_784', version=1, cache=True ) #, return_X_y=False)

X, y = mnist['data'] / 255., mnist['target']