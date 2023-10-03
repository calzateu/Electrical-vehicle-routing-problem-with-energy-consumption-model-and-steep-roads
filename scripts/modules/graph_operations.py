import osmnx as ox


def search_places_inside_graph(places, graph, threshold = 50):
    # Within the graph, find the nearest node to the actual location of our places
    places_within_graph = {}

    for key in places.keys():
        # Nearest node to the coordinate
        coord = places[key]
        node, dist = ox.nearest_nodes(graph, X=coord[1], Y=coord[0], return_dist=True)

        # Verify if the node is near the point
        if dist <= threshold: # in meters, maximum allowed distance
            places_within_graph[key] = [node, dist]
        else:
            print(f"The place '{key}' is {dist:.2f} meters away from the graph, "
                  f"which is more than {threshold:.2f} meters away")

    return places_within_graph
