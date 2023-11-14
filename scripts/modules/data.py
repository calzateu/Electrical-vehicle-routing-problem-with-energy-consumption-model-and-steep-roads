from collections import defaultdict

class Data:
    def __init__(self, number_of_nodes, number_of_vehicles, capacity_of_vehicles,
                 max_energy, dist_matrix, demands):
        self.number_of_nodes = number_of_nodes
        self.number_of_vehicles = number_of_vehicles
        self.capacity_of_vehicles = capacity_of_vehicles
        self.max_energy = max_energy
        self.dist_matrix = dist_matrix
        self.demands = demands
