import math
import matplotlib.pyplot as plt

from .model import energy_between_a_b


def initialize_graph_information(graph):
    # Calculate edge elevations
    for u, v, k, data in graph.edges(keys=True, data=True):
        start_elev = graph.nodes[u]['elevation']
        end_elev = graph.nodes[v]['elevation']
        length = data['length']
        elevation = data['elevation'] = (start_elev + end_elev) / 2
        angle = math.atan2(end_elev - start_elev, length)
        angle = math.degrees(angle)
        data['angle'] = angle
        print(
            f"The edge ({u}, {v}) has an elevation angle of {angle} degrees and an elevation of {elevation} meters"
        )


def get_graph_information(graph):
    # Return angle and distances

    angles = dict()
    distances = dict()

    for u, v, k, data in graph.edges(keys=True, data=True):
        angles[(u, v)] = float(data['angle'])
        distances[(u, v)] = float(data['length'])

    return angles, distances


def create_cost_graph(graph, angles, distances, information):
    # Calculate edge costs
    cost_graph = dict()
    for u, v, k, data in graph.edges(keys=True, data=True):
        cost_graph[(u, v)] = energy_between_a_b(angles[(u, v)], distances[(u, v)], information)
        if angles[(u, v)] >= 0 > cost_graph[(u, v)]:
            print("################## Warning ##################")
            print(angles[(u, v)], distances[(u, v)], cost_graph[(u, v)])

    return cost_graph


def analyze_problem(graph, path, distances):
    elevations = []
    distances_path = [0]
    for i in range(1, len(path)):
        elevations.append(graph.nodes[path[i]]['elevation'])
        distances_path.append(distances_path[-1] + distances[(path[i - 1], path[i])])

    distances_path.pop()

    plt.plot(distances_path, elevations)
    plt.show()


def bellman_ford(cost_graph, start_node, end_node, distances):
    # Extract nodes from edge keys and initialize them with zero weight.
    nodes = set()
    for edge in cost_graph.keys():
        source, destination = edge
        nodes.add(source)
        nodes.add(destination)

    # Initialize the node weights and paths dictionaries
    distances_bellman = {node: float('inf') for node in nodes}
    paths = {node: [] for node in nodes}
    distances_bellman[start_node] = 0
    paths[start_node] = [start_node]

    # Relax the edges V-1 times (V is the total of nodes)
    num_nodes = len(set(edge for edge, _ in cost_graph.items()))
    for _ in range(num_nodes - 1):
        for edge, weight in cost_graph.items():
            source, destination = edge
            new_distance = distances_bellman[source] + weight
            if new_distance < distances_bellman[destination]:
                distances_bellman[destination] = new_distance
                paths[destination] = paths[source] + [destination]

    # Check for negative cycles
    for edge, weight in cost_graph.items():
        source, destination = edge
        new_distance = distances_bellman[source] + weight
        if new_distance < distances_bellman[destination]:
            print(f"source: {source}, destination: {destination}")
            print(f"distance: {distances_bellman[destination]}")
            print(f"new_distance: {new_distance}")
            analyze_problem(cost_graph, paths[destination], distances)
            print(paths[destination])
            raise ValueError("The graph contains a negative cycle")

    # Get the path between the points of interest
    shortest_path = paths[end_node]

    # Get the distance between the points of interest
    min_distance = distances_bellman[end_node]

    return shortest_path, min_distance


def create_customer_graph(places_within_graph, distances, cost_graph):
    dict_cost_matrix = dict()
    dict_paths = dict()
    values = places_within_graph.values()
    it = 0
    for value1 in values:
        for value2 in values:
            print(it)
            a = value1[0]
            b = value2[0]
            if a != b:
                minimum_path, cost = bellman_ford(cost_graph, a, b, distances)
                dict_cost_matrix[(a, b)] = cost
                dict_paths[(a, b)] = minimum_path
            else:
                dict_cost_matrix[(a, b)] = 0
                dict_paths[(a, b)] = [0]
            it += 1

    return dict_cost_matrix, dict_paths


# Compare the min distance path and the min cost path
def path_cost(path, angles, distances, information):
    cost = 0
    for i in range(1, len(path)):
        cost += energy_between_a_b(angles[(path[i-1], path[i])], distances[(path[i-1], path[i])], information)

    return cost


# Specify the names of the functions to export
__all__ = ["initialize_graph_information", "get_graph_information", "create_cost_graph",
           "analyze_problem", "bellman_ford", "create_customer_graph", "path_cost"]
