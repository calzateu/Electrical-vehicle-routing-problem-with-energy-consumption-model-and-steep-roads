import osmnx as ox
import math
import matplotlib.pyplot as plt

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

points = [tuple(coord) for coord in polygon_places.values()]

places = ['El Poblado, Medellin, Colombia']

# Use retain_all to keep all disconnected sub-graphs (e.g., if your places aren't contiguous)
GP = ox.graph_from_place(places, network_type="drive", retain_all=True)
fig, ax = ox.plot_graph(GP, edge_color="#FFFF5C", edge_linewidth=0.25, node_color='red', show=False)

# Plot the additional points with a different color
x, y = zip(*points)
ax.scatter(y, x, c='blue', marker='o')  # Change 'blue' to the desired color
plt.show()

# Add elevations to each node and calculate angles
# **Important: Make sure to have a .tif file in the folder!!**
G = ox.add_node_elevations_raster(GP, '../../elevation_data/10s090w_20101117_gmted_mea075.tif')

# Calculate edge elevations
for u, v, k, data in G.edges(keys=True, data=True):
    start_elev = G.nodes[u]['elevation']
    end_elev = G.nodes[v]['elevation']
    length = data['length']
    data['elevation'] = (start_elev + end_elev) / 2

# Calculate edge elevation angles
angles = dict()
for u, v, k, data in G.edges(keys=True, data=True):
    start_elev = G.nodes[u]['elevation']
    end_elev = G.nodes[v]['elevation']
    length = data['length']
    elevation = data['elevation']
    angle = math.atan2(end_elev - start_elev, length)
    angle = math.degrees(angle)
    angles[(u, v)] = angle
    print(f"The edge ({u}, {v}) has an elevation angle of {angle} degrees and an elevation of {elevation} meters")

# Histogram of elevation angles
data = angles.values()
counts, edges, bars = plt.hist(data)
plt.bar_label(bars)
plt.title("Elevation Angle Histogram")
plt.xlabel("Angles")
plt.ylabel("Count")
plt.show()

# Scatter plot of elevation angles
plt.plot(data, 'o')
plt.title("Elevation Angle Scatter Plot")
plt.xlabel("Sequential Number")
plt.ylabel("Angles")
plt.show()

print("The max angle is: ", max(data))
print("The min angle is: ", min(data))

ox.save_graphml(G, 'el_poblado_graph.graphml')
