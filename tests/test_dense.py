import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

from kviz.dense import DenseGraph


def test_dense_no_input():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    dg = DenseGraph(model)
    dg.render(filename='test_no_input')
    g1 = dg.get_graph()
    dg.set_graph(g1)
    dg.get_graph()


def test_dense_input_xor():
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

    model.fit(X, Y, batch_size=4, epochs=1000)

    colors = np.array(['b', 'g'])
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color=colors[Y].tolist(), s=50, alpha=0.8)

    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    meshData = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(meshData)
    Z = np.array([0 if x < .5 else 1 for x in Z])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired)
    ax.axis('off')

    fig.savefig("test_model_xor.png")

    dg = DenseGraph(model)
    dg.animate_activations(X, filename='test_input_xor')


def test_dense_input_line():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    t, _ = datasets.make_blobs(n_samples=50, centers=[[.5, .5]], cluster_std=.1, random_state=1)
    X = np.array(t)
    Y = np.array([1 if x[0] - x[1] >= 0 else 0 for x in X])

    model.fit(X, Y, batch_size=50, epochs=100)

    # see which nodes activate for a given class
    X0 = X[X[:, 0] - X[:, 1] <= 0]
    X1 = X[X[:, 0] - X[:, 1] >= 0]

    X = np.concatenate((X0, X1), axis=0)

    fig, ax = plt.subplots()
    ax.scatter(X0[:, 0], X0[:, 1], color='b', s=10, alpha=0.8)
    ax.scatter(X1[:, 0], X1[:, 1], facecolors='none', edgecolors='black')

    h = .01
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    meshData = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(meshData)
    Z = np.array([0 if x < .5 else 1 for x in Z])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired)
    ax.axis('off')

    fig.savefig("test_model_line.png")

    dg = DenseGraph(model)
    dg.animate_activations(X, filename='test_input_line', duration=300)


def test_animate_learning():
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

    dg = DenseGraph(model)
    dg.animate_learning(X, Y, filename='test_animate', verbose=0, batch_size=50)


def test_activated_by():
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

    dg = DenseGraph(model)
    dg.animate_neuron_activated_by(X, Y, 0, 1, filename="test_activated_by", duration=100, epochs=1500, batch_size=100)
