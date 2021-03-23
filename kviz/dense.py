"""
CopyRight goes here
"""

import json
import pygraphviz
from networkx import DiGraph, set_node_attributes
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation

class DenseGraph():

    def __init__(self, model):
        self.model = model
        self._graph = self._make_digraph()


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
                    graph.add_node(str(l) + str(n), shape='circle', style='filled', color='#3498db', fontcolor='#3498db')
                else:
                    graph.add_node(str(l) + str(n), color='#2ecc71', style='filled', fontcolor='#2ecc71', shape='circle')
    
                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        graph.add_node(str(l+1) + str(h), shape='circle', style='filled', color='#3498db', fontcolor='#3498db')
                    graph.add_edge(str(l) + str(n), str(l+1) + str(h), color='#B20000')
        
        return graph


    def render(self, filename='graph'):
        """
        Render visualization of a Sequential Dense keras model.
        """
        out = to_agraph(self._graph)
        out.layout(prog='dot')
        out.draw(filename+'.png')
        return
