"""
Copyright 2021 Lance Galletti

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
from matplotlib.colors import rgb2hex, Normalize
from networkx import DiGraph, set_node_attributes
from networkx.drawing.nx_agraph import to_agraph
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from kviz.helper_functions import (
    get_or_create_colormap_with_dict, unique_index
)


class DenseGraph():
    """
        Class for creating and rendering visualization of Keras
        Sequential models with only dense layers

        Attributes:
            model : tf.keras.Model
                a compiled keras sequential model with only dense layers

        Methods:
            get_graph : returns nx.DiGraph
                DiGraph is computed based on the model provided
                you can access this attribute using this method

            set_graph : takes nx.DiGraph
                DiGraph is computed based on the model provided
                you can set this attribute to a modified DiGraph
                using this method

            render :
                Can print the network architecture or, if input
                is provided, show a GIF of the activations of each
                Neuron based on the input provided.

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


    def _make_digraph(self):
        """
        Constructs the DiGraph
        """
        graph = DiGraph(nodesep='1', ranksep='1')

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]

            if type(layer) != Dense:
                raise ValueError("Unsupported Layer Type: %s Only Dense Layers are Supported", type(layer))

            for n in range(0, layer.input_shape[1]):
                if l == 0:
                    graph.add_node(
                        unique_index(l, n),
                        shape="circle",
                        color="#3498db",
                        label=''
                    )
                else:
                    graph.add_node(
                        unique_index(l, n),
                        shape="circle",
                        color="#2ecc71",
                        label=''
                    )

                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        graph.add_node(
                            unique_index(l + 1, h),
                            shape="circle",
                            color="#3498db",
                            label=''
                        )
                    graph.add_edge(
                        unique_index(l, n),
                        unique_index(l + 1, h),
                        color="#B20000"
                    )

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
                int_layer = self.model.layers[prev_layer]
                int_model.add(Dense(
                    int_layer.output_shape[1],
                    input_dim=int_layer.input_shape[1],
                    activation=int_layer.activation)
                )
                int_model.layers[prev_layer].set_weights(int_layer.get_weights())
            int_model.compile(loss=self.model.loss)
            intermediate_models.append(int_model)

        return intermediate_models


    def _snap(self, filename):
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


    def _snap_learning(self, X, Y, filename):
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

        # TODO catch max number of classes
        colors = np.array([x for x in 'bgrcmyk'])

        fig, ax = plt.subplots(frameon=False)
        ax.scatter(X[:, 0], X[:, 1], color=colors[Y].tolist(), s=100, alpha=.9)
        Z = self.model.predict(meshData)
        Z = np.array([int(round(z[0])) for z in Z]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.5, cmap=plt.cm.Paired)
        fig.savefig(filename + '.png')
        plt.close()

        return np.asarray(im.open(filename + '.png'))


    def _snap_activated_by(self, X, neuron_layer, neuron_node, filename='activated_by'):
        """
        Take snapshot of input with decision boundary of the data that activates a particular neuron

        Parameters:
            X : ndarray
                input to a Keras model
            neuron_layer : int
                the layer in which the neuron is located
            neuron_node : int
                the location of the neuron in the layer
            filename : str
                name of file to which visualization will be saved

        Returns:
            ndarray of the image
        """
        activated_by = []
        predictions = self._int_models[neuron_layer].predict(X)

        for i in range(len(X)):
            if predictions[i][neuron_node] > 0:
                activated_by.append(i)

        return self._snap_X(activated_by, X, filename)


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
            filename + '_stacked.gif',
            optimize=False,  # important for transparent background
            save_all=True,
            append_images=stacked_imgs[1:],
            loop=0,
            duration=duration
        )

        return


    def _reset(self):
        """
        Resets the graph labels, colors, fonts
        """
        self._graph = self._graph_original_copy.copy()


    def animate_learning(self, X, Y, snap_freq=10, filename='learn', duration=1000, **kwargs):
        """
        Make GIF from snapshots of decision boundary at given snap_freq

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
            images.append(im.fromarray(self._snap_learning(X, Y, filename)))

        images[0].save(
            filename + '.gif',
            optimize=False,  # important for transparent background
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration
        )
        self._int_models = self._get_int_models()
        return self.model


    def animate_activations(self, X, filename='activations', duration=1000, x_color="#3498db", x_marker="o"):
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

        network_images = []
        input_images = []

        predictions = [X]
        for i in range(len(self._int_models)):
            predictions.append(self._int_models[i].predict(X))
        predictions.append(self.model.predict(X))

        for i in range(len(X)):
            for l in range(len(self.model.layers)):
                layer = self.model.layers[l]

                layerVals = predictions[l][i]
                norm = Normalize(vmin=min(layerVals), vmax=max(layerVals))

                color_maps = {}

                for n in range(0, layer.input_shape[1]):
                    act = predictions[l][i][n]

                    index = unique_index(l, n)
                    the_color_map = get_or_create_colormap_with_dict(self._graph.nodes[index]["color"], color_maps)

                    if l == 0:
                        set_node_attributes(self._graph, {
                            index: {
                                'style': 'filled',
                                'color': str(rgb2hex(the_color_map(norm(act))))
                            }})
                        if int(act) == act:
                            set_node_attributes(self._graph, {
                                index: {
                                    'label': str(act)
                                }})
                    else:
                        set_node_attributes(self._graph, {
                            index: {
                                'style': 'filled',
                                'color': str(rgb2hex(the_color_map(norm(act))))
                            }})

                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        act = predictions[l + 1][i][h]

                        index = unique_index(l + 1, h)
                        the_color_map = get_or_create_colormap_with_dict(self._graph.nodes[index]["color"], color_maps)

                        set_node_attributes(self._graph, {
                            index: {
                                'label': str(int(round(act))),
                                'style': 'filled',
                                'color': str(rgb2hex(the_color_map(norm(act))))
                            }})

            network_images.append(self._snap(filename))
            input_images.append(self._snap_X([i], X, filename, x_color=x_color, x_marker=x_marker))
            self._reset()

        self._stack_gifs(network_images, input_images, filename, duration=duration)
        return


    def animate_neuron_activated_by(
            self, X, Y,
            neuron_layer,
            neuron_node,
            filename='animate_activated_by',
            snap_freq=10, duration=1000, **kwargs):
        """
        Creates a visualization of the data that activates a particular neuron through the learning process

        Parameters:
            X : ndarray
                input to a Keras model
            Y : ndarray
                classes to be learned
            neuron_layer : int
                the layer in which the neuron is located
            neuron_node : int
                the location of the neuron in the layer
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

        activated_by = []
        decision_boundary = []
        if 'epochs' in kwargs:
            epochs = kwargs['epochs']
            kwargs.pop('epochs', None)
        else:
            epochs = snap_freq

        for _ in range(int(epochs / snap_freq)):
            # TODO: once this is learned you can't visualize the same for other neurons
            self.model.fit(X, Y, epochs=snap_freq, **kwargs)
            self._int_models = self._get_int_models()  # re-compute intermediate models TODO make this more efficient
            activated_by.append(self._snap_activated_by(X, neuron_layer, neuron_node, filename + 'act'))
            decision_boundary.append(self._snap_learning(X, Y, filename + 'db'))

        self._stack_gifs(activated_by, decision_boundary, filename, duration=duration)
        return self.model


    def render(self, filename='graph'):
        """
        Creates a visualization of the graph for a Sequential Dense keras model

        Parameters:
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

        self._snap(filename)
        return
