import osmnx as ox
import json
import matplotlib.pyplot as plt

from modules.graphics import plot_places
from modules.model import *
from modules.costs_and_paths import *

# Load extracted data
graph = ox.load_graphml('el_poblado_graph.graphml')

# Get elevation angles and distances between nodes
angles, distances = get_graph_information(graph)

# Points of interest
# El tesoro
# ISA
# Amsterdam
# Mall Zona 2
# Mall la visitación
# Clínica El Rosario
# La Vaquita (los balsos)
# Euro (la inferior)
# Mall del este
# Complex los balsos

polygon_places = {
    'Complex Los Balsos': [6.186898127000028, -75.56143294740268],
    'Mall Del Este': [6.198756717661721, -75.556393638567],
    'Amsterdam Plaza': [6.202176172201083, -75.55510765722],
    'Euro Supermercados La Inferior': [6.199854351194308, -75.56472210973091],
    'Mall Zona 2': [6.1985640740178996, -75.56511452001308],
    'Mall La Visitación': [6.196673041613839, -75.56516852042641],
    'Complex Los Balsos 2': [6.186898127000028, -75.56143294740268],
}

# Data Exploration
# El Poblado Graph
plot_places(graph, polygon_places)

# Problem Preparation
# Within the graph, find the nearest node to the actual location of our places
threshold = 50  # in meters, maximum allowed distance
places_within_graph = {}

for key in polygon_places.keys():
    # Nearest node to the coordinate
    coord = polygon_places[key]
    node, dist = ox.nearest_nodes(graph, X=coord[1], Y=coord[0], return_dist=True)

    # Verify if the node is near the point
    if dist <= threshold:
        places_within_graph[key] = [node, dist]
    else:
        print(f"The place '{key}' is {dist:.2f} meters away from the graph, "
              f"which is more than {threshold:.2f} meters away")

# Our point of interest along with the coordinates of the nearest node within the graph
print(places_within_graph)

# Finding the nearest routes between two nodes
los_balsos = places_within_graph['Complex Los Balsos']
euro = places_within_graph['Euro Supermercados La Inferior']
print("########### Shortest path between Complex Los Balsos and Euro Supermercados La Inferior ###########")
print(ox.shortest_path(graph, los_balsos[0], euro[0]))


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

# # Comparison of energy consumption between different points with different angles

# ## Energy consumption between two nodes with an angle of 0
# Two arbitrary nodes
a_0 = 3791874410
b_0 = 5476066477

angle_0, distance_0, energy_0, energy_per_meter_0 = energy_distance_and_energy_per_meter(
    a_0, b_0, angles, distances, information
)

# ### Energy consumption between the two nodes with the highest elevation angle (71.57°)
a_max, b_max = max(angles, key=angles.get)

angle_max, distance_max, energy_max, energy_per_meter_max = energy_distance_and_energy_per_meter(
    a_max, b_max, angles, distances, information
)

# ## Energy consumption between the two nodes with the lowest elevation angle (-63.28°)
a_min, b_min = min(angles, key=angles.get)

angle_min, distance_min, energy_min, energy_per_meter_min = energy_distance_and_energy_per_meter(
    a_min, b_min, angles, distances, information
)

# ## Comparison of results
print('{:<18} {:<20} {:<28} {:<28}'.format('Angle', 'Distance', 'Energy Consumed', 'Energy per Meter'))
print('{:<18} {:<20} {:<28} {:<28}'.format(angle_0, distance_0, energy_0, energy_per_meter_0))
print('{:<18} {:<20} {:<28} {:<28}'.format(angle_max, distance_max, energy_max, energy_per_meter_max))
print('{:<18} {:<20} {:<28} {:<28}'.format(angle_min, distance_min, energy_min, energy_per_meter_min))

# # Comparison of Energy Consumption Between Different Points with Different Angles

# ## Energy Consumption Between Two Nodes with 0° Angle
# Two arbitrary nodes
a_0 = 3791874410
b_0 = 5476066477

angles[(a_0, b_0)] = -angles[(a_0, b_0)]
angle_0, distance_0, energy_0, energy_per_meter_0 = energy_distance_and_energy_per_meter(
    a_0, b_0, angles, distances, information
)

# ### Energy Consumption Between the Two Nodes with the Highest Elevation Angle (71.57°)
a_max, b_max = max(angles, key=angles.get)

angles[(a_max, b_max)] = -angles[(a_max, b_max)]
angle_max, distance_max, energy_max, energy_per_meter_max = energy_distance_and_energy_per_meter(
    a_max, b_max, angles, distances, information
)
angles[(a_max, b_max)] = -angles[(a_max, b_max)]

# ## Energy Consumption Between the Two Nodes with the Lowest Elevation Angle (-63.28°)
a_min, b_min = min(angles, key=angles.get)

angles[(a_min, b_min)] = -angles[(a_min, b_min)]
angle_min, distance_min, energy_min, energy_per_meter_min = energy_distance_and_energy_per_meter(
    a_min, b_min, angles, distances, information
)
angles[(a_min, b_min)] = -angles[(a_min, b_min)]

# ## Comparison of Results
print('{:<18} {:<20} {:<28} {:<28}'.format('Angle', 'Distance', 'Consumed Energy', 'Energy per Meter'))
print('{:<18} {:<20} {:<28} {:<28}'.format(angle_0, distance_0, energy_0, energy_per_meter_0))
print('{:<18} {:<20} {:<28} {:<28}'.format(angle_max, distance_max, energy_max, energy_per_meter_max))
print('{:<18} {:<20} {:<28} {:<28}'.format(angle_min, distance_min, energy_min, energy_per_meter_min))

# # Create Cost Graph to Travel from One Point of Interest to Another
# Calculate edge costs
cost_graph = dict()
for u, v, k, data in graph.edges(keys=True, data=True):
    cost_graph[(u, v)] = energy_between_a_b(angles[(u, v)], distances[(u, v)], information)
    if angles[(u, v)] >= 0 > cost_graph[(u, v)]:
        print("################## Warning ##################")
        print(angles[(u, v)], distances[(u, v)], cost_graph[(u, v)])


def analyze_problem(path, distances):
    elevations = []
    distances_path = [0]
    for i in range(1, len(path)):
        elevations.append(graph.nodes[path[i]]['elevation'])
        distances_path.append(distances_path[-1] + distances[(path[i - 1], path[i])])

    distances_path.pop()

    plt.plot(distances_path, elevations)
    plt.show()


def bellman_ford(graph, start_node, end_node, distances):
    # Extract nodes from edge keys and initialize them with zero weight
    nodes = set()
    for edge in graph.keys():
        source, destination = edge
        nodes.add(source)
        nodes.add(destination)

    # Initialize the node weights and paths dictionaries
    distances_bellman = {node: float('inf') for node in nodes}
    paths = {node: [] for node in nodes}
    distances_bellman[start_node] = 0
    paths[start_node] = [start_node]

    # Relax the edges V-1 times (V is the number of nodes)
    num_nodes = len(set(edge for edge, _ in graph.items()))
    for _ in range(num_nodes - 1):
        for edge, weight in graph.items():
            source, destination = edge
            new_distance = distances_bellman[source] + weight
            if new_distance < distances_bellman[destination]:
                distances_bellman[destination] = new_distance
                paths[destination] = paths[source] + [destination]

    # Check for negative cycles
    for edge, weight in graph.items():
        source, destination = edge
        new_distance = distances_bellman[source] + weight
        if new_distance < distances_bellman[destination]:
            print(f"source: {source}, destination: {destination}")
            print(f"distance: {distances_bellman[destination]}")
            print(f"new_distance: {new_distance}")
            analyze_problem(paths[destination], distances)
            print(paths[destination])
            raise ValueError("The graph contains a negative cycle")

    # Get the path between the points of interest
    shortest_path = paths[end_node]

    # Get the distance between the points of interest
    min_distance = distances_bellman[end_node]

    return shortest_path, min_distance


def create_dist_distances(lugares_dentro_grafo, distances):
    dict_cost_matrix = dict()
    dict_paths = dict()
    values = lugares_dentro_grafo.values()
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


dict_cost_matrix, dict_paths = create_dist_distances(places_within_graph, distances)


# ## Compare the min distance path and the min cost path
def path_cost(path, distances, information):
    cost = 0
    for i in range(1, len(path)):
        cost += energy_between_a_b(angles[(path[i-1], path[i])], distances[(path[i-1], path[i])], information)

    return cost


def select_minimum_path(paths, distances, information):
    minimum_path = []
    minimum_cost = float('inf')

    for path in paths:
        cost = path_cost(path, distances, information)

        if cost < minimum_cost:
            minimum_cost = cost
            minimum_path = path

    return minimum_path, minimum_cost


place1 = places_within_graph['Euro Supermercados La Inferior']
place2 = places_within_graph['Complex Los Balsos']

paths = ox.k_shortest_paths(graph, place1[0], place2[0], 10)
min_distance_path = ox.shortest_path(graph, place1[0], place2[0])

print("Minimum Distance Path", path_cost(min_distance_path, distances, information))

path, cost = select_minimum_path(paths, distances, information)
print("Cost", cost)

# Plot altitudes along the route
elevations_min_distance = []
distances_min_distance = [0]
for i in range(1, len(min_distance_path)):
    elevations_min_distance.append(graph.nodes[min_distance_path[i]]['elevation'])
    distances_min_distance.append(
        distances_min_distance[-1] + distances[(min_distance_path[i-1], min_distance_path[i])]
    )

distances_min_distance.pop()

elevations_min_cost = []
distances_min_cost = [0]
for i in range(1, len(path)):
    elevations_min_cost.append(graph.nodes[path[i]]['elevation'])
    distances_min_cost.append(distances_min_cost[-1] + distances[(path[i-1], path[i])])

distances_min_cost.pop()

plt.plot(distances_min_distance, elevations_min_distance)
plt.show()

plt.plot(distances_min_cost, elevations_min_cost)
plt.show()

total_distance_min_distance = 0
for i in range(1, len(min_distance_path)):
    total_distance_min_distance += distances[(min_distance_path[i-1], min_distance_path[i])]

total_distance_min_cost = 0
for i in range(1, len(path)):
    total_distance_min_cost += distances[(path[i-1], path[i])]

print("Distance of the minimum distance route", total_distance_min_distance)
print("Distance of the minimum consumption route", total_distance_min_cost)

# Saving cost graph
# It is necessary because to save the dictionaries in JSON, we have to have keys in string format
dict_cost_matrix_output = dict((str(k), v) for k, v in dict_cost_matrix.items())
dict_paths_output = dict((str(k), v) for k, v in dict_paths.items())

print(dict_cost_matrix_output)

with open("polygon_places.json", "w") as outfile:
    json.dump(polygon_places, outfile)

with open("places_within_graph.json", "w") as outfile:
    json.dump(places_within_graph, outfile)

with open("dict_cost_matrix.json", "w") as outfile:
    json.dump(dict_cost_matrix_output, outfile)

with open("dict_paths.json", "w") as outfile:
    json.dump(dict_paths_output, outfile)
