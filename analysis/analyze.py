""" main analysis and plot script """


from itertools import combinations
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from scipy.stats import sem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import pingouin as pg
import seaborn as sns
import scipy.stats as stats

from indimap import IndiMap
from indimap.util import map_func, stat_func
import util.dataset as dataset


def manage_path():
    """ Create graph path for plots created with this script """

    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs'
    graph_path.mkdir(parents=True, exist_ok=True)
    return graph_path


def init_map():
    """ Initialize IndiMap objects for all models and datasets """
    return [
        # IndiMap(dataset.get_rtnet_on_mnist()),
        # IndiMap(dataset.get_alexnet_on_mnist()),
        # IndiMap(dataset.get_resnet18_on_mnist()),
        IndiMap(dataset.get_rtnet_on_ecoset10()),
        IndiMap(dataset.get_alexnet_on_ecoset10()),
        IndiMap(dataset.get_resnet18_on_ecoset10()),
    ]


def compute(all_maps, load=False):
    """ Compute/load all results """
    for obj in all_maps:
        print(obj)
        obj.compute_pred(load_exists=False)

def main():
    graph_path = manage_path()
    all_maps = init_map()
    compute(all_maps, load=True)
    # graph(all_maps, graph_path)


if __name__ == "__main__":
    main()