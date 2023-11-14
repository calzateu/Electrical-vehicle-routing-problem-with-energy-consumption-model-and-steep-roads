from collections import defaultdict

class Data:
    def __init__(self, number_of_nodes, number_of_vehicles, capacity_of_vehicles,
                 max_energy, dist_matrix, demands, reference_nodes, node_to_index,
                 angles, distances, information):
        self.number_of_nodes = number_of_nodes
        self.number_of_vehicles = number_of_vehicles
        self.capacity_of_vehicles = capacity_of_vehicles
        self.max_energy = max_energy
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.reference_nodes = reference_nodes
        self.node_to_index = node_to_index
        self.angles = angles
        self.distances = distances
        self.information = information
