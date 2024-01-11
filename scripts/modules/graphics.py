import osmnx as ox
import matplotlib.pyplot as plt


def plot_places(graph, polygon_places):
    # Data Exploration
    # El Poblado Graph
    fig, ax = ox.plot_graph(graph, edge_color="#FFFF5C", edge_linewidth=0.25, node_color='red', show=False)

    # Graph with points of interest in blue color
    points = [tuple(coord) for coord in polygon_places.values()]
    x, y = zip(*points)
    ax.scatter(y, x, c='blue', marker='o')  # Change 'blue' to the desired color
    plt.show()


def plot_histogram_angles(angles):
    # Histogram of elevation angles
    data = angles.values()
    counts, edges, bars = plt.hist(data)
    plt.bar_label(bars)
    plt.title("Elevation Angle Histogram")
    plt.xlabel("Angles")
    plt.ylabel("Count")
    plt.show()


def plot_scatter_angles(angles):
    # Scatter plot of elevation angles
    data = angles.values()
    plt.plot(data, 'o')
    plt.title("Elevation Angle Scatter Plot")
    plt.xlabel("Sequential Number")
    plt.ylabel("Angles")
    plt.show()


# Specify the names of the functions to export
__all__ = ["plot_places", "plot_histogram_angles", "plot_scatter_angles"]
