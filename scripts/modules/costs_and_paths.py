import math


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
