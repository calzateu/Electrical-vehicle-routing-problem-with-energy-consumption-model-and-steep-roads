import numpy as np
import osmnx as ox

from .costs_and_paths import path_cost
from .data import Data
from .constructive4 import ConstructiveMethod4


def build_dist_matrix(nodes, number_of_nodes, customer_graph):
    """"""
    dist_matrix = np.zeros((number_of_nodes, number_of_nodes))
    node_to_index = dict()
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 != node2:
                dist_matrix[i, j] = customer_graph[(node1, node2)]

        node_to_index[i] = node1

    return dist_matrix, node_to_index


def print_paths_cost(paths, customer_graph_paths, node_to_index, angles, distances, information):
    """"""
    for path in paths:
        original_path = []
        for i in range(len(path) - 1):
            nodes_temp = customer_graph_paths[(node_to_index[path[i]], node_to_index[path[i + 1]])]
            original_path.extend(nodes_temp[:-1])

        print(path_cost(original_path, angles, distances, information))


def build_customer_graph_distances(graph, nodes, distances):
    """"""
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

    return customer_graph_distances, customer_graph_paths_distances


def find_paths(nodes, number_of_nodes, customer_graph, number_of_vehicles, capacity_of_vehicles,
               max_energy, demands, reference_elements, angles, distances, information, customer_graph_paths,
               **kwargs):
    """"""
    dist_matrix, node_to_index = build_dist_matrix(nodes, number_of_nodes, customer_graph)

    print(dist_matrix)

    data = Data(number_of_nodes, number_of_vehicles, capacity_of_vehicles, max_energy, dist_matrix,
                demands.copy(), reference_elements, node_to_index, angles, distances, information.copy())

    # constructive = ConstructiveMethod3(data)
    constructive = ConstructiveMethod4(data)

    paths = constructive.search_paths()

    print(paths)
    print()

    comparative_customer_graph_paths = customer_graph_paths
    if kwargs.get("customer_graph_paths_reference", 0) != 0:
        comparative_customer_graph_paths = kwargs["customer_graph_paths_reference"]

    print_paths_cost(paths, comparative_customer_graph_paths, node_to_index, angles, distances, information)

    print()
