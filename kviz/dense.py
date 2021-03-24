"""
CopyRight goes here
"""

import json
import imageio
import pygraphviz
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, Normalize
from networkx import DiGraph, set_node_attributes
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation

class DenseGraph():

    def __init__(self, model):
        self.model = model
        self._graph = self._make_digraph()
        self._int_models = self._get_int_models()


    def _make_digraph(self):
        """
        Constructs the DiGraph
        """

        graph = DiGraph(bgcolor="transparent", nodesep='1', ranksep='1')
        prev_layer = None

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]

            if type(layer) != Dense:
                raise ValueError("Unsupported Layer Type: %s Only Dense Layers are Supported", type(layer))
            
            for n in range(0, layer.input_shape[1]):
                if l == 0:
                    graph.add_node(str(l) + str(n), shape='circle', color='#3498db', label='')
                else:
                    graph.add_node(str(l) + str(n), shape='circle', color='#2ecc71', label='')
    
                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        graph.add_node(str(l+1) + str(h), shape='circle', color='#3498db', label='')
                    graph.add_edge(str(l) + str(n), str(l+1) + str(h), color='#B20000')
        
        return graph


    def _get_int_models(self):
        intermediate_models = []

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]
            int_model = keras.Sequential()

            for prev_layer in range(l+1):
                int_layer = self.model.layers[prev_layer]
                int_model.add(Dense(int_layer.output_shape[1], input_dim=int_layer.input_shape[1], activation=int_layer.activation))  
                int_model.layers[prev_layer].set_weights(int_layer.get_weights())
            int_model.compile(loss=self.model.loss)
            intermediate_models.append(int_model)
        
        return intermediate_models


    def _snap(self, filename):
        out = to_agraph(self._graph)
        out.layout(prog='dot')
        out.draw(filename+'.png')
        return np.asarray(im.open(filename+'.png'))


    def _snap_input(self, i, input, filename):
        fig, ax = plt.subplots()
        ax.scatter(input[:,0], input[:,1], s=300, facecolors='none', edgecolors='#3498db')
        ax.scatter(input[i,0], input[i,1], s=300, color='#3498db')
        ax.spines['bottom'].set_color('#3498db')
        ax.spines['left'].set_color('#3498db')
        ax.tick_params(axis='x', colors='#3498db')
        ax.tick_params(axis='y', colors='#3498db')
        fig.savefig(filename+'_input.png', transparent=True)
        return np.asarray(im.open(filename+'_input.png'))

    
    def _stack_gifs(self, imgs1, imgs2, filename, duration):
        stacked_imgs = []
        for i in range(len(imgs1)):
            img1 = imgs1[i]
            img2 = imgs2[i]
            wpercent = img2.shape[0] / img2.shape[1]
            proportional_height = int(img1.shape[1] * wpercent)
            im1 = im.fromarray(img1)
            im2 = im.fromarray(img2)
            img2 = np.asarray(im2.resize((img1.shape[1], proportional_height)))

            stacked = im.fromarray(np.vstack((np.asarray(img2), np.asarray(img1))))
            stacked_imgs.append(stacked)

        stacked_imgs[0].save(
                filename+'_stacked.gif',
                optimize=False,
                save_all=True,
                append_images=stacked_imgs[1:],
                loop=0,
                duration=duration*1000,
                transparency=255,
                disposal=2
            )
        
        return


    def _reset(self):
        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]
            for n in range(0, layer.input_shape[1]):
                if l == 0:
                    set_node_attributes(self._graph, {str(l) + str(n):{'label': '', 'fontcolor': '', 'style': '', 'color': '#3498db'}})
                else:
                    set_node_attributes(self._graph, {str(l) + str(n):{'label': '', 'fontcolor': '', 'style': '', 'color': '#2ecc71'}})
            
            for h in range(0, layer.output_shape[1]):
                if l == len(self.model.layers) - 1:
                    set_node_attributes(self._graph, {str(l+1) + str(h):{'label': '', 'fontcolor': '', 'style': '', 'color': '#3498db'}})


    def render(self, input=None, filename='graph', duration=1):
        """
        Render visualization of a Sequential Dense keras model.
        """
        
        if input is None:
            self._snap(filename)
            return
        
        network_images = []
        input_images = []

        norm = Normalize(vmin=-1, vmax=1.1)
        gcmap = plt.cm.Greens
        bcmap = plt.cm.Blues

        predictions = [input]
        for i in range(len(self._int_models)):
            predictions.append(self._int_models[i].predict(input))
        predictions.append(self.model.predict(input))

        for i in range(len(input)):
            for l in range(len(self.model.layers)):
                layer = self.model.layers[l]
                int_model = self._int_models[l]

                for n in range(0, layer.input_shape[1]):
                    act = predictions[l][i][n]
                    if l == 0:
                        set_node_attributes(self._graph, {str(l) + str(n):{'style': 'filled', 'color': str(rgb2hex(bcmap(norm(act))))}})
                        if int(act) == act:
                            set_node_attributes(self._graph, {str(l) + str(n):{'label': str(act)}})
                    else:
                        set_node_attributes(self._graph, {str(l) + str(n):{'style': 'filled', 'color': str(rgb2hex(gcmap(norm(act))))}})

                if l == len(self.model.layers) - 1:
                    network_images.append(self._snap(filename))
                    input_images.append(self._snap_input(i, input, filename))
                    self._reset()
                    
                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        act = predictions[l][i][h]
                        set_node_attributes(self._graph, {str(l+1) + str(h):{'label': str(int(round(act))), 'style': 'filled', 'color': str(rgb2hex(bcmap(norm(act))))}})

                network_images.append(self._snap(filename))
                input_images.append(self._snap_input(i, input, filename))
                self._reset()
            self._reset()
        
        self._stack_gifs(network_images, input_images, filename, duration=duration)
        return
        