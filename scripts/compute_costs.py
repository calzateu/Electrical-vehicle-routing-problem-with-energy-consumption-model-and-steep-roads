import json
import matplotlib.pyplot as plt

from modules.graphics import plot_places
from modules.model import *
from modules.costs_and_paths import *
from modules.graph_operations import *

import constants

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
# Within the graph, find the nearest node to the actual location of our places.
threshold = 50  # in meters, maximum allowed distance
places_within_graph = search_places_inside_graph(places=polygon_places, graph=graph, threshold=threshold)

# Finding the nearest routes between two nodes
los_balsos = places_within_graph['Complex Los Balsos']
euro = places_within_graph['Euro Supermercados La Inferior']
print("########### Shortest path between Complex Los Balsos and Euro Supermercados La Inferior ###########")
print(ox.shortest_path(graph, los_balsos[0], euro[0]))


information = constants.constants_dict.copy()

# # Comparison of energy consumption between different points with different angles

# ## Energy consumption between two nodes with an angle of 0
# Two arbitrary nodes.
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
# Two arbitrary nodes.
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

# # Create Cost Graph to Travel from One Point to Another
# Calculate edge costs.
cost_graph = create_cost_graph(graph=graph, angles=angles, distances=distances, information=information)


customer_graph, customer_graph_paths = create_customer_graph(places_within_graph, distances, cost_graph)


# ## Compare the min distance path and the min cost path.
place1 = places_within_graph['Euro Supermercados La Inferior']
place2 = places_within_graph['Complex Los Balsos']

min_distance_path = ox.shortest_path(graph, place1[0], place2[0])
min_cost_path, min_cost_distance = bellman_ford(cost_graph, place1[0], place2[0], distances)


print("Minimum Distance Path", path_cost(min_distance_path, angles, distances, information))
print("Minimum Cost Path", path_cost(min_cost_path, angles, distances, information))

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
for i in range(1, len(min_cost_path)):
    elevations_min_cost.append(graph.nodes[min_cost_path[i]]['elevation'])
    distances_min_cost.append(distances_min_cost[-1] + distances[(min_cost_path[i-1], min_cost_path[i])])

distances_min_cost.pop()

plt.plot(distances_min_distance, elevations_min_distance)
plt.show()

plt.plot(distances_min_cost, elevations_min_cost)
plt.show()

total_distance_min_distance = 0
for i in range(1, len(min_distance_path)):
    total_distance_min_distance += distances[(min_distance_path[i-1], min_distance_path[i])]

total_distance_min_cost = 0
for i in range(1, len(min_cost_path)):
    total_distance_min_cost += distances[(min_cost_path[i-1], min_cost_path[i])]

print("Distance of the minimum distance route", total_distance_min_distance)
print("Distance of the minimum cost route", total_distance_min_cost)

# Saving cost graph
# It is necessary because to save the dictionaries in JSON, we have to have keys in string format.
customer_graph_output = dict((str(k), v) for k, v in customer_graph.items())
customer_graph_paths_output = dict((str(k), v) for k, v in customer_graph_paths.items())
cost_graph_output = dict((str(k), v) for k, v in cost_graph.items())

print(customer_graph_output)

with open("polygon_places.json", "w") as outfile:
    json.dump(polygon_places, outfile)

with open("places_within_graph.json", "w") as outfile:
    json.dump(places_within_graph, outfile)

with open("customer_graph.json", "w") as outfile:
    json.dump(customer_graph_output, outfile)

with open("customer_graph_paths.json", "w") as outfile:
    json.dump(customer_graph_paths_output, outfile)

with open("cost_graph.json", "w") as outfile:
    json.dump(cost_graph_output, outfile)
