from modules.graphics import *
from modules.costs_and_paths import *

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

# Example of location on the map
location = ox.geocode("Centro Comercial El Tesoro, Medellín, Colombia")
latitude = location[0]
longitude = location[1]
print("Latitude for El Tesoro Mall:", latitude)
print("Longitude for El Tesoro Mall:", longitude)

places = ['El Poblado, Medellin, Colombia']

# Use retain_all to keep all disconnected sub-graphs (e.g., if your places aren't contiguous)
graph = ox.graph_from_place(places, network_type="drive", retain_all=True)

# Data Exploration
# El Poblado Graph
plot_places(graph, polygon_places)

# Add elevations to each node and calculate angles
# **Important: Make sure to have a .tif file in the folder!!**
graph = ox.add_node_elevations_raster(graph, '../../elevation_data/10s090w_20101117_gmted_mea075.tif')

initialize_graph_information(graph)

angles, distances = get_graph_information(graph)

# Histogram of elevation angles
plot_histogram_angles(angles)

# Scatter plot of elevation angles
plot_scatter_angles(angles)


data = angles.values()
print("The max angle is: ", max(data))
print("The min angle is: ", min(data))

ox.save_graphml(graph, 'el_poblado_graph.graphml')
