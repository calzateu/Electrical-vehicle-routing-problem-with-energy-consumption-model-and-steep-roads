import json
import ast

from modules.cost_matrix_operations import *
from modules.costs_and_paths import *
from modules.heuristic_tools import *

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

# Iteración 1
find_paths(nodes, number_of_nodes, customer_graph, number_of_vehicles, capacity_of_vehicles, max_energy,
           demands, last_two_elements, angles, distances, information, customer_graph_paths)

# Iteración 2
customer_graph_distances, customer_graph_paths_distances = build_customer_graph_distances(graph, nodes, distances)

find_paths(nodes, number_of_nodes, customer_graph_distances, number_of_vehicles, capacity_of_vehicles, max_energy,
           demands, last_two_elements, angles, distances, information, customer_graph_paths_distances,
           customer_graph_paths_reference=customer_graph_paths)
