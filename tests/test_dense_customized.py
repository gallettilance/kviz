import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from networkx import set_node_attributes, set_edge_attributes

from kviz.dense import DenseGraph
from kviz.helper_functions import get_random_color, get_random_shape

# for testing dense models with different colors and shapes. Most of the codes are from test_viz.py.


def test_dense_input_xor_customized():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2, activation=ACTIVATION))
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
    fig.savefig("test_model_xor_customized.png")
    dg = DenseGraph(model)

    the_graph = dg.get_graph()
    for l in range(len(model.layers)):
        layer = model.layers[l]

        for n in range(0, layer.input_shape[1]):
            set_node_attributes(the_graph, {
                str(l) + str(n): {
                    'shape': "diamond",
                    'color': "#00ff00",
                    'label': ""
                }
            })

            for h in range(0, layer.output_shape[1]):
                if l == len(model.layers) - 1:
                    set_node_attributes(the_graph, {
                        str(l + 1) + str(h): {
                            'shape': "square",
                            'color': "#ff0000",
                            'label': ""
                        }
                    })
                set_edge_attributes(the_graph, {
                    (str(l) + str(n), str(l + 1) + str(h)): {
                        'color': "#0000ff"
                    }
                })

    dg.set_graph(the_graph)
    dg.render(X, filename='test_input_xor_customized', x_color="#FF0000", x_marker="^")


def test_dense_input_line_customized():
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

    fig.savefig("test_model_line_customized.png")

    dg = DenseGraph(model)
    # test: each node is different
    the_graph = dg.get_graph()
    for l in range(len(model.layers)):
        layer = model.layers[l]

        for n in range(0, layer.input_shape[1]):
            set_node_attributes(the_graph, {
                str(l) + str(n): {
                    'shape': get_random_shape(),
                    'color': get_random_color(),
                    'label': ""
                }
            })

            for h in range(0, layer.output_shape[1]):
                if l == len(model.layers) - 1:
                    set_node_attributes(the_graph, {
                        str(l + 1) + str(h): {
                            'shape': get_random_shape(),
                            'color': get_random_color(),
                            'label': ""
                        }
                    })
                set_edge_attributes(the_graph, {
                    (str(l) + str(n), str(l + 1) + str(h)): {
                        'color': get_random_color()
                    }
                })

    dg.set_graph(the_graph)

    dg.render(X, filename='test_input_line_customized', duration=300)
