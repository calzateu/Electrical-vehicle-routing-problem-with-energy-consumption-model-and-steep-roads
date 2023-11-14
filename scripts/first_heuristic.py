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


# The following parameters are based on the article, except for the efficiency information.
# Problem and vehicle parameters
v_i = 0
v_f = 0
C_r = 0.0064
C_d = 0.7
A = 8.0
rho = 1.2
g = 9.81

# Input parameters
v_ab = 6  # Desired speed per segment
m = 1000
acceleration = 0.8  # From the paper
deceleration = -0.9  # From the paper
# deceleration = -0.8  # My own


# Efficiency information. Search references or make a regression analysis (as they did)
eta_acceleration_positive = 0.8
eta_acceleration_negative = 1.9  # Critical point 1.8
eta_constant_positive = 0.8
eta_constant_negative = 1.9
eta_deceleration_positive = 0.8
eta_deceleration_negative = 1.9

information = initialize_information(v_i, v_f, v_ab, acceleration, deceleration,
                                     m, C_r, C_d, A, rho, g, eta_acceleration_positive,
                                     eta_acceleration_negative, eta_constant_positive,
                                     eta_constant_negative, eta_deceleration_positive,
                                     eta_deceleration_negative)



first_two_elements, last_two_elements, elements_with_min_difference = select_reference_points_inside_matrix(
    cost_graph=cost_graph
)

nodes = [value[0] for value in places_within_graph.values()]
number_of_nodes = len(nodes) # I am taking the first place as the depot
number_of_vehicles = 1
capacity_of_vehicles = 2000
max_energy = 10000
#dist_matrix = cost_graph
#demands = [11, 4, 6, 12, 9, 8, 13]
demands = [0, 400, 600, 1200, 900, 800, 1300]

dist_matrix = np.zeros((number_of_nodes, number_of_nodes))
node_to_index = dict()
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if node1 != node2:
            dist_matrix[i, j] = customer_graph[(node1, node2)]

    node_to_index[i] = node1


data = Data(number_of_nodes, number_of_vehicles, capacity_of_vehicles, max_energy, dist_matrix,
            demands, last_two_elements, node_to_index, angles, distances, information)

#place_and_demand = dict()
#for key, value in places_within_graph.items():
#    place_and_demand[value[0]] = demands.pop()


#constructive = ConstructiveMethod3(data)
constructive = ConstructiveMethod4(data)

paths = constructive.search_paths()

print(paths)


