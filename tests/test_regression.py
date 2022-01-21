import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from kviz.dense import DenseGraph


def test_regression_3dx():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ])
    Y = np.array([x[0] + x[1] + x[2] for x in X])

    ACTIVATION = "relu"
    model = keras.models.Sequential()
    model.add(layers.Dense(4, input_shape=(3, ), activation=ACTIVATION))
    model.add(layers.Dense(4, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))

    optimizer = keras.optimizers.RMSprop(0.05)

    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])

    model.fit(X, Y, epochs=100)

    dg = DenseGraph(model)
    dg.animate_regression(X, filename="test_regression_3dx")


def test_regression_2dx():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    Y = np.array([2 * x[0] + 3 * x[1] + 4 for x in X])

    ACTIVATION = "relu"
    model = keras.models.Sequential()
    model.add(layers.Dense(4, input_shape=(2, ), activation=ACTIVATION))
    model.add(layers.Dense(4, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))

    optimizer = keras.optimizers.RMSprop(0.03)

    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])

    model.fit(X, Y, epochs=100)

    dg = DenseGraph(model)
    dg.animate_regression(X, filename="test_regression_2dx")


def test_regression_1dx():
    X = np.array([[2], [4], [6], [8]])
    Y = np.array([2 * x[0] + 1 for x in X])

    ACTIVATION = "relu"
    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_shape=(1, ), activation=ACTIVATION))
    model.add(layers.Dense(3, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))

    optimizer = keras.optimizers.RMSprop(0.025)

    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])

    model.fit(X, Y, epochs=50)

    dg = DenseGraph(model)
    dg.animate_regression(X, filename="test_regression_1dx")
