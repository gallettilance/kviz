import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.datasets as datasets

from kviz.visualizer import Visualizer


def test_render():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    dg = Visualizer(model)
    dg.render(filename='test_render')
    g1 = dg.get_graph()
    dg.set_graph(g1)


def test_view_activations_for():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])
    Y = np.array([x[0] ^ x[1] for x in X])

    dg = Visualizer(model)
    dg.fit(X, Y, 10, 'test_view_activations_for_decision_boundary', 100, epochs=2000)
    dg.view_activations_for(X, filename='test_view_activations_for')


def test_fit():
    ACTIVATION = "sigmoid"

    def custom_activation(x):
        return x**2

    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=custom_activation))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    t, _ = datasets.make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=1, random_state=2)
    X = np.array(list(filter(lambda x: x[0]**2 + x[1]**2 < 1 or x[0]**2 + x[1]**2 > 1.5, t)))
    Y = np.array([1 if x[0]**2 + x[1]**2 >= 1 else 0 for x in X])

    dg = Visualizer(model)
    dg.fit(X, Y, 10, 'test_fit', 100, epochs=200, verbose=0, batch_size=50)
    dg.view_activations_for(X, 'test_fit_activations', 100)
