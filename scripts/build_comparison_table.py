import json
import osmnx as ox
import ast
import numpy as np

from modules.cost_matrix_operations import *
from modules.data import Data
from modules.constructive3 import ConstructiveMethod3
from modules.constructive4 import ConstructiveMethod4
from modules.costs_and_paths import *
from modules.model import *

with open("polygon_places.json", "r") as infile:
    polygon_places = json.load(infile)

with open("places_within_graph.json", "r") as infile:
    places_within_graph = json.load(infile)

with open("customer_graph.json", "r") as infile:
    customer_graph = json.load(infile)

customer_graph = {ast.literal_eval(key): value for key, value in customer_graph.items()}

with open("customer_graph_paths.json", "r") as infile:
    customer_graph_paths = json.load(infile)

customer_graph_paths = {ast.literal_eval(key): value for key, value in customer_graph_paths.items()}

with open("cost_graph.json", "r") as infile:
    cost_graph = json.load(infile)

cost_graph = {ast.literal_eval(key): value for key, value in cost_graph.items()}


graph = ox.load_graphml('el_poblado_graph.graphml')
# Get elevation angles and distances between nodes
angles, distances = get_graph_information(graph)



