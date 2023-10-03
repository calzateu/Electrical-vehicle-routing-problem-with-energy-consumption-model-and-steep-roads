import json
import osmnx as ox


with open("polygon_places.json", "r") as infile:
    polygon_places = json.load(infile)

with open("places_within_graph.json", "r") as infile:
    places_within_graph = json.load(infile)

with open("customer_graph.json", "r") as infile:
    customer_graph = json.load(infile)

with open("customer_graph_paths.json", "r") as infile:
    customer_graph_paths = json.load(infile)

with open("cost_graph.json", "r") as infile:
    cost_graph = json.load(infile)

graph = ox.load_graphml('el_poblado_graph.graphml')


print(dict(sorted(cost_graph.items(), key=lambda item: item[1])))
