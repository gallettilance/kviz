"""
Copyright 2024 Lance Galletti

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import numpy as np
from PIL import Image as im

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, LinearSegmentedColormap

from networkx import DiGraph, set_node_attributes
from networkx.drawing.nx_agraph import to_agraph

import tensorflow.keras as keras


COLORS = np.array(['purple', 'blue'])
tuples = list(zip(
    map(plt.Normalize(0, 1), [0, .5, 1]),
    [COLORS[0], 'white', COLORS[1]])
)
CMAP = LinearSegmentedColormap.from_list("", tuples, 100)


class Visualizer():
    """
        Class for creating and rendering visualization of Keras
        Sequential models with only dense layers

        Attributes:
            model : tf.keras.Model
                a compiled keras sequential model with only dense layers

        Methods:
            render :
                Creates a Graph visualization of the model

            fit :
                Make GIF from snapshots of decision boundary at
                given intervals of epochs during training

            view_activations_for :
                Creates a GIF of the activations of each
                Neuron for the input provided
    """

    def __init__(self, model):
        """
        Parameters:
            model: A keras model.
                Since our class is called DenseGraph(), the keras model should only contain dense layers.
        """
        self.model = model
        self._graph = self._make_digraph()
        self._graph_original_copy = self._graph.copy()
        self._int_models = self._get_int_models()


    def get_graph(self):
        """
        DiGraph is computed based on the model provided \
        You can access this attribute using this method

        Returns:
            networkx.DiGraph

        """
        return self._graph


    def set_graph(self, graph=DiGraph()):
        """
        DiGraph is computed based on the model provided \
        you can set this attribute to a modified DiGraph \
        using this method

        Parameters:
            graph : networkx.Graph
                The graph to set

        Returns:
            None
        """
        self._graph = graph
        self._graph_original_copy = self._graph.copy()
        return


    def _unique_index(self, layer, node):
        return str(layer) + "_" + str(node)


    def _make_digraph(self):
        """
        Constructs the DiGraph
        """
        graph = DiGraph(nodesep='1', ranksep='1')

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]

            if type(layer) is not keras.layers.Dense:
                raise ValueError("Unsupported Layer Type: %s Only Dense Layers are Supported", type(layer))

            if l == 0:
                input_size = layer._input_shape_arg[0]
                for i in range(input_size):
                    graph.add_node(
                        self._unique_index(l, i),
                        shape="circle",
                        color="#3498db",
                        label=''
                    )
            output_size = layer.units

            for o in range(output_size):
                if l == len(self.model.layers) - 1:
                    graph.add_node(
                        self._unique_index(l + 1, o),
                        shape="circle",
                        color="#3498db",
                        label=''
                    )
                else:
                    graph.add_node(
                        self._unique_index(l + 1, o),
                        shape="circle",
                        color="#2ecc71",
                        label=''
                    )
                for i in range(input_size):
                    graph.add_edge(
                        self._unique_index(l, i),
                        self._unique_index(l + 1, o),
                        color="#B20000"
                    )
            input_size = layer.units
        return graph


    def _get_int_models(self):
        """
        Creates a list of models where each model consists of all the layers
        (with matching weights and activations) in self.model up to a specified
        layer depth. This is important to get each node activation.
        """
        intermediate_models = []

        for l in range(len(self.model.layers)):
            int_model = keras.Sequential()
            for prev_layer in range(l + 1):
                int_model.add(self.model.layers[prev_layer])
            int_model.compile(loss=self.model.loss)
            intermediate_models.append(int_model)

        return intermediate_models


    def _snap_graph(self, filename):
        """
        Take snapshot image of the graph
        """
        out = to_agraph(self._graph)
        out.layout(prog='dot')
        out.draw(filename + '.png')
        return np.asarray(im.open(filename + '.png'))


    def _snap_X(self, indexes, X, filename, x_color="#3498db", x_marker="o"):
        """
        Take snapshot image of the input

        TODO:
            1. this doesn't work for X in > 2 dimension
            2. how to plot input could / should be specified by user

        Parameters:
            indexes: list of int.
                list of the indexes of the point in X that should be bold
            X: List.
                a list of coordinates of the points
            filename: str.
                name of file to which visualization will be saved
            x_color: str.
                the color (in hex form) of the points in the pyplot graph
            x_marker: str.
                the shape of the points in the pyplot graph
        """
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], s=300, marker=x_marker, facecolors='none', edgecolors=x_color)
        for i in indexes:
            ax.scatter(X[i, 0], X[i, 1], s=300, marker=x_marker, color=x_color)
        ax.spines['bottom'].set_color(x_color)
        ax.spines['left'].set_color(x_color)
        ax.tick_params(axis='x', colors=x_color)
        ax.tick_params(axis='y', colors=x_color)
        fig.savefig(filename + '_X.png')
        plt.close()
        return np.asarray(im.open(filename + '_X.png'))


    def _snap_decision_boundary(self, X, Y, filename):
        """
        Take snapshot of input with decision boundary
        """
        # create a mesh to plot in
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        meshData = np.c_[xx.ravel(), yy.ravel()]

        fig, ax = plt.subplots(frameon=False)
        ax.scatter(X[:, 0], X[:, 1], color=COLORS[Y].tolist(), s=100, alpha=.9)
        Z = self.model.predict(meshData).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.4, cmap=CMAP)
        fig.savefig(filename + '.png')
        plt.close()

        return np.asarray(im.open(filename + '.png'))


    def _snap_regression(self, X, Y, filename):
        """
        Take snapshot of the regression line
        """
        x_min, x_max = X.min() - .5, X.max() + .5
        y_min, y_max = Y.min() - .5, Y.max() + .5
        xplot = np.linspace(x_min, x_max, 200)
        fig, ax = plt.subplots(frameon=False)
        ax.scatter(X, Y, color=COLORS[1], s=40, alpha=.7)
        ax.plot(xplot, self.model.predict(xplot.reshape(X.shape)), linewidth=2, color=COLORS[0])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        fig.savefig(filename + '.png')
        plt.close()

        return np.asarray(im.open(filename + '.png'))


    def _stack_gifs(self, imgs1, imgs2, filename, duration):
        """
        Takes two lists of images and stacks each image in one list on top
        of its corresponding image in the other. Then creates a GIF of the
        list of stacked images.
        """
        stacked_imgs = []
        for i in range(len(imgs1)):
            img1 = imgs1[i]
            img2 = imgs2[i]
            wpercent = img2.shape[0] / img2.shape[1]
            proportional_height = int(img1.shape[1] * wpercent)
            im2 = im.fromarray(img2)
            img2 = np.asarray(im2.resize((img1.shape[1], proportional_height)))

            stacked = im.fromarray(np.vstack((np.asarray(img2), np.asarray(img1))))
            stacked_imgs.append(stacked)

        stacked_imgs[0].save(
            filename + '.gif',
            optimize=False,  # important for transparent background
            save_all=True,
            append_images=stacked_imgs[1:],
            loop=0,
            duration=duration
        )

        return


    def _convert_gif(self, imgs, filename, duration):
        """
        Convert a list of images to a gif.

        Args:
            imgs: List.
                a list of images
            filename: str.
                the filename for the gif
            duration: int
                duration in ms between images in GIF

        Returns:
            None

        """
        stacked_images = []
        for img in imgs:
            stacked_images.append(im.fromarray(np.asarray(img)))

        stacked_images[0].save(
            filename + '.gif',
            optimize=False,
            save_all=True,
            append_images=stacked_images[1:],
            loop=0,
            duration=duration
        )

        return


    def _reset(self):
        """
        Resets the graph labels, colors, fonts
        """
        self._graph = self._graph_original_copy.copy()


    def fit(self, X, Y, snap_freq=10, filename='decision_boundary', duration=1000, **kwargs):
        """
        Make GIF from snapshots of decision boundary at given snap_freq of epochs during training

        Parameters:
            X : ndarray
                input to a Keras model
            Y : ndarray
                classes to be learned
            snap_freq : int
                number of epochs after which to take a snapshot
            filename : str
                name of file to save as GIF
            duration : int
                duration in ms between images in GIF
            **kwargs : other params
                paramter inputs to model.fit

        Returns:
            The model after learning
        """

        images = []
        if 'epochs' in kwargs:
            epochs = kwargs['epochs']
            kwargs.pop('epochs', None)
        else:
            epochs = snap_freq

        for _ in range(int(epochs / snap_freq)):
            self.model.fit(X, Y, epochs=snap_freq, **kwargs)
            self._int_models = self._get_int_models()  # TODO: make this function more efficient

            if self.model.loss == 'binary_crossentropy':
                images.append(im.fromarray(self._snap_decision_boundary(X, Y, filename)))
            if self.model.loss == 'mean_squared_error':
                images.append(im.fromarray(self._snap_regression(X, Y, filename)))

        self._convert_gif(images, filename, duration)
        return self.model


    def view_activations_for(self, X, filename='activations', duration=1000, x_color="#3498db", x_marker="o"):
        """
        Creates an animation of the graph activated by each data point

        Parameters:
            X : ndarray
                input to a Keras model
            filename : str
                name of file to which visualization will be saved
            duration : int
                duration in ms between images in GIF
            x_color: str.
                the color (in hex form) of the points in the pyplot graph
            x_marker: str.
                the shape of the points in the pyplot graph

        Returns:
            None
        """

        cvals = [0, 1]
        colors = ['white', 'green']
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        CMAP_GREEN = LinearSegmentedColormap.from_list("", tuples, 100)

        network_images = []
        input_images = []

        predictions = [X]
        for i in range(len(self._int_models)):
            predictions.append(self._int_models[i].predict(X))
        predictions.append(self.model.predict(X))

        for i in range(len(X)):
            input_size = self.model.layers[0]._input_shape_arg[0]
            for l in range(len(self.model.layers)):
                if l != 0:
                    input_size = self.model.layers[l - 1].units

                for n in range(0, input_size):
                    act = predictions[l][i][n]
                    index = self._unique_index(l, n)
                    if l == 0:
                        set_node_attributes(self._graph, {
                            index: {
                                'label': "+" if act > 0 else "-"
                            }})
                    else:
                        set_node_attributes(self._graph, {
                            index: {
                                'style': 'filled',
                                'color': str(rgb2hex(CMAP_GREEN(act)))
                            }})

                for h in range(0, self.model.layers[l].units):
                    if l == len(self.model.layers) - 1:
                        act = predictions[l + 1][i][h]
                        index = self._unique_index(l + 1, h)

                        set_node_attributes(self._graph, {
                            index: {
                                'style': 'filled',
                                'color': str(rgb2hex(CMAP(act)))
                            }})

            network_images.append(self._snap_graph(filename))
            input_images.append(self._snap_X([i], X, filename, x_color=x_color, x_marker=x_marker))
            self._reset()

        self._stack_gifs(network_images, input_images, filename, duration)
        return


    def render(self, filename='graph'):
        """
        Creates a visualization of the graph for a Sequential Dense keras model

        Parameters:
            filename : str
                name of file to which visualization will be saved

        Returns:
            None
        """

        self._snap_graph(filename)
        return
