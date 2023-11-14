import json
import osmnx as ox
import ast
import numpy as np

from modules.cost_matrix_operations import *
from modules.data import Data
from modules.constructive3 import ConstructiveMethod3

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

first_two_elements, last_two_elements, elements_with_min_difference = select_reference_points_inside_matrix(
    cost_graph=cost_graph
)

nodes = [value[0] for value in places_within_graph.values()]
number_of_nodes = len(nodes) # I am taking the first place as the depot
number_of_vehicles = 1
capacity_of_vehicles = 20
max_energy = 10000
#dist_matrix = cost_graph
#demands = [11, 4, 6, 12, 9, 8, 13]
demands = [0, 4, 6, 12, 9, 8, 13]

dist_matrix = np.zeros((number_of_nodes, number_of_nodes))
node_to_index = dict()
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if node1 != node2:
            dist_matrix[i, j] = customer_graph[(node1, node2)]

    node_to_index[i] = node1


data = Data(number_of_nodes, number_of_vehicles, capacity_of_vehicles, max_energy, dist_matrix, demands)

#place_and_demand = dict()
#for key, value in places_within_graph.items():
#    place_and_demand[value[0]] = demands.pop()


constructive = ConstructiveMethod3(data)

paths = constructive.search_paths()

print(paths)


