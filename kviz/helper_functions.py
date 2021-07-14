from matplotlib.colors import ListedColormap
import numpy as np
import re
import logging


# According to https://graphviz.org/doc/info/shapes.html.
# There are three main types of shapes : polygon-based, record-based and user-defined.
# For now, this list only supports some polygon-based shapes.
valid_networkx_shapes = [
    "box", "polygon", "ellipse", "oval", "circle", "egg", "triangle", "diamond", "trapezium",
    "parallelogram", "house", "pentagon", "hexagon", "septagon", "octagon", "doublecircle", "doubleoctagon",
    "tripleoctagon", "invtriangle", "invtrapezium", "invhouse", "Mdiamond", "Msquare", "Mcircle", "rect", "rectangle",
    "square", "star", "cylinder",
]


def create_colormap(hex_color_string, N=25, step=51):
    """

        Parameters:
            hex_color_string: str.
                A string in format "#ffffff".
            N: int.
                should be within [1, 256]. The bigger N is, the more color the colormap will contain.
            step: int.
                Controls the range of the color map; the bigger step is, the bigger the range.
                A step that is either too big or too small might cause problems.

        Returns: a matplotlib colormap

    """
    hex_color_string = hex_color_string.lstrip('#')
    r, g, b = tuple(int(hex_color_string[i: i + 2], 16) for i in (0, 2, 4))

    left_r = max(0, r - step)
    right_r = min(255, r + step)
    left_g = max(0, g - step)
    right_g = min(255, g + step)
    left_b = max(0, b - step)
    right_b = min(255, b + step)

    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(left_r / right_r, 1, N)
    vals[:, 1] = np.linspace(left_g / right_g, 1, N)
    vals[:, 2] = np.linspace(left_b / right_b, 1, N)
    newcmp = ListedColormap(vals)
    return newcmp


def check_regular_exp(input_color, inner_color, output_color,
                      edge_clr, input_shape, inner_shape, output_shape, silence):
    """
        The function to check whether the inputs are valid.

        Parameters:
            input_color: str.
                The color of the input layer in hex form (e.g. "#FFFFFF").
            inner_color: str.
                The color of the inner layer(s) in hex form.
            output_color: str.
                The color of the output layer in hex form.
            edge_clr: str.
                The color of the edge connecting nodes of different layer. In hex form.
            input_shape: str.
                The shape of the nodes in the input layer. Should be a valid shape (e.g. "polygon").
                Check https://graphviz.org/doc/info/shapes.html for some valid shapes.
                Note that now only polygon-based shapes are supported.
            inner_shape: str.
                The shape of the nodes in the inner layer(s). Should be a valid shape.
            output_shape: str.
                The shape of the nodes in the output layer. Should be a valid shape.
            silence: bool.
                Default is True. If set to False, an exception will raise if at least one input does not meet the
                regular expression.

        Returns: bool.
            True if all inputs are valid.

    """
    expression = r'^#(?:[0-9a-fA-F]{3}){1,2}$'
    if re.search(expression, input_color) and re.search(expression, inner_color) and \
            re.search(expression, output_color) and re.search(expression, edge_clr) and \
            inner_shape in valid_networkx_shapes and input_shape in valid_networkx_shapes and \
            output_shape in valid_networkx_shapes:
        return True
    else:
        if silence:
            logging.warning("At least one of the inputs regarding colors&shapes is invalid."
                            "\nsilence=True.\nUse default settings instead.")
            return False
        else:
            logging.warning("At least one of the inputs regarding colors&shapes is invalid."
                            "\nsilence=False.\nRaise an error.")
            raise ValueError
