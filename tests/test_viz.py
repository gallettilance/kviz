import pytest
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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
    g2 = dg.get_graph()


def test_dense_input_xor():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]])
    Y = np.array([x[0]^x[1] for x in X])

    history = model.fit(X, Y, batch_size=4, epochs=1000)

    colors = np.array(['b', 'g'])
    fig, ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1], color=colors[Y].tolist(), s=50, alpha=0.8)

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
    dg.render(input=X, filename='test_input_xor')


def test_dense_input_line():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    X = np.array([
        [ 0.541655, 0.69532781],
        [ 0.80655584, 0.79387175],
        [ 0.5021    , 0.8571741 ],
        [ 0.36550791, 0.42808937],
        [ 0.39783897, 0.26387356],
        [ 0.47935623, 0.5821197 ],
        [ 0.64581811, 0.52579658],
        [ 0.32244285, 0.10384071],
        [ 0.73575591, 0.46401503],
        [ 0.56261354, 0.32918085],
        [ 0.32090669, 0.5773805 ],
        [ 0.43041757, 0.53126979],
        [ 0.8735116 , 0.30454442],
        [ 0.42253464, 0.43953945],
        [ 0.49436355, 0.58566637],
        [ 0.28584948, 0.71089035],
        [ 0.24944093, 0.65549807],
        [ 0.29028941, 0.21599641],
        [ 0.51330344, 0.56049438],
        [ 0.79881581, 0.45896835],
        [ 0.17396033, 0.59255645],
        [ 0.85281047, 0.58003144],
        [ 0.32584057, 0.38423007],
        [ 0.52880871, 0.7908547 ],
        [ 0.39806956, 0.41238514],
        [ 0.58046833, 0.36303798],
        [ 0.57127328, 0.64131463],
        [ 0.95395092, 0.20912687],
        [ 0.52538242, 0.58039787],
        [ 0.43768949, 0.51123307],
        [ 0.26697003, 0.6801653 ],
        [ 0.69001768, 0.46972856],
        [ 0.31854033, 0.51038908],
        [ 0.79765044, 0.87917784],
        [ 0.41936461, 0.74448901],
        [ 0.6957476 , 0.94817864],
        [ 0.37313558, 0.42745177],
        [ 0.67288724, 0.351567  ],
        [ 0.59313249, 0.19275126],
        [ 0.74605814, 0.74047597],
        [ 0.53098949, 0.5756325 ],
        [ 0.17722043, 0.45745194],
        [ 0.58877265, 0.56673487],
        [ 0.5091517 , 0.46256323],
        [ 0.33737074, 0.15474348],
        [ 0.15874596, 0.89015508],
        [ 0.65220755, 0.524335  ],
        [ 0.72788014, 0.25303484],
        [ 0.53548523, 0.41964381]])
    Y = np.array([1 if x[0] - x[1] >= 0 else 0 for x in X])

    history = model.fit(X, Y, batch_size=49, epochs=100)

    # see which nodes activate for a given class
    X0 = X[X[:,0] - X[:,1] <= 0]
    X1 = X[X[:,0] - X[:,1] >= 0]

    X = np.concatenate((X0, X1), axis=0)

    fig, ax = plt.subplots()
    ax.scatter(X0[:,0],X0[:,1], color='b', s=10, alpha=0.8)
    ax.scatter(X1[:,0],X1[:,1], facecolors='none', edgecolors='black')

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
    dg.render(input=X, filename='test_input_line', duration=.3)
