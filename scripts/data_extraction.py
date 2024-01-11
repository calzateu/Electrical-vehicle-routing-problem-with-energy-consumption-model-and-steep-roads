import osmnx as ox
from modules.graphics import *
from modules.costs_and_paths import *

# Coordinates of points of interest
polygon_places = {
    'Complex Los Balsos': [6.186898127000028, -75.56143294740268],
    'Mall Del Este': [6.198756717661721, -75.556393638567],
    'Amsterdam Plaza': [6.202176172201083, -75.55510765722],
    'Euro Supermarkets La Inferior': [6.199854351194308, -75.56472210973091],
    'Mall Zone 2': [6.1985640740178996, -75.56511452001308],
    'Mall La Visitación': [6.196673041613839, -75.56516852042641],
    'Complex Los Balsos 2': [6.186898127000028, -75.56143294740268],
}

# Example of location on the map
location = ox.geocode("Centro Comercial El Tesoro, Medellín, Colombia")
latitude, longitude = location
print("Latitude for El Tesoro Mall:", latitude)
print("Longitude for El Tesoro Mall:", longitude)

# List of points of interest
places = ['El Poblado, Medellin, Colombia']

# Get the road network graph for the area
graph = ox.graph_from_place(places, network_type="drive", retain_all=True)

# Data Exploration | Graph of the area and elevations
plot_places(graph, polygon_places)

# Add elevations to each node and calculate angles
graph = ox.add_node_elevations_raster(graph, '../../elevation_data/10s090w_20101117_gmted_mea075.tif')

initialize_graph_information(graph)

angles, distances = get_graph_information(graph)

# Histogram of elevation angles
plot_histogram_angles(angles)

# Scatter plot of elevation angles
plot_scatter_angles(angles)

# Print elevation angle statistics
print("The max angle is: ", max(angles))
print("The min angle is: ", min(angles))

# Save the graph in GraphML format
ox.save_graphml(graph, 'el_poblado_graph.graphml')
