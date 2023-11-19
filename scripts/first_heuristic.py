import json
import osmnx as ox
import ast
import numpy as np

from modules.cost_matrix_operations import *
from modules.data import Data
from modules.constructive4 import ConstructiveMethod4
from modules.costs_and_paths import *

import constants

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

information = constants.constants_dict.copy()

first_two_elements, last_two_elements, elements_with_min_difference = select_reference_points_inside_matrix(
    cost_graph=cost_graph
)

nodes = [value[0] for value in places_within_graph.values()]
number_of_nodes = len(nodes)  # I am taking the first place as the depot
number_of_vehicles = 1
capacity_of_vehicles = 1000
max_energy = 10000
demands = [0, 200, 300, 600, 450, 400, 650]

dist_matrix = np.zeros((number_of_nodes, number_of_nodes))
node_to_index = dict()
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if node1 != node2:
            dist_matrix[i, j] = customer_graph[(node1, node2)]

    node_to_index[i] = node1

data = Data(number_of_nodes, number_of_vehicles, capacity_of_vehicles, max_energy, dist_matrix,
            demands.copy(), last_two_elements, node_to_index, angles, distances, information.copy())

# place_and_demand = dict()
# for key, value in places_within_graph.items():
#    place_and_demand[value[0]] = demands.pop()


# constructive = ConstructiveMethod3(data)
constructive = ConstructiveMethod4(data)

paths = constructive.search_paths()

print(paths)
print()

for path in paths:
    original_path = []
    for i in range(len(path) - 1):
        nodes_temp = customer_graph_paths[(node_to_index[path[i]], node_to_index[path[i + 1]])]
        original_path.extend(nodes_temp[:-1])

    print(path_cost(original_path, angles, distances, information))

print()

customer_graph_distances = dict()
customer_graph_paths_distances = dict()
for node1 in nodes:
    for node2 in nodes:
        if node1 != node2:
            path_temp = ox.shortest_path(graph, node1, node2)
            customer_graph_paths_distances[(node1, node2)] = path_temp
            distance = 0
            for i in range(len(path_temp) - 1):
                distance += distances[(path_temp[i], path_temp[i + 1])]

            customer_graph_distances[(node1, node2)] = distance

        else:
            customer_graph_paths_distances[(node1, node2)] = [node1]
            customer_graph_distances[(node1, node2)] = 0

print(customer_graph_distances)

dist_matrix = np.zeros((number_of_nodes, number_of_nodes))
node_to_index = dict()
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if node1 != node2:
            dist_matrix[i, j] = customer_graph_distances[(node1, node2)]

    node_to_index[i] = node1

print(dist_matrix)

data = Data(number_of_nodes, number_of_vehicles, capacity_of_vehicles, max_energy, dist_matrix,
            demands.copy(), last_two_elements, node_to_index, angles, distances, information.copy())

constructive = ConstructiveMethod4(data)

paths = constructive.search_paths()

print(paths)
print()

for path in paths:
    original_path = []
    for i in range(len(path) - 1):
        nodes_temp = customer_graph_paths_distances[(node_to_index[path[i]], node_to_index[path[i + 1]])]
        original_path.extend(nodes_temp[:-1])

    print(path_cost(original_path, angles, distances, information))
