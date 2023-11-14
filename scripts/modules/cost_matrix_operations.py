

def select_reference_points_inside_matrix(cost_graph):

    # Step 1: Sort the dictionary by values in ascending order
    sorted_cost_graph_by_value = dict(sorted(cost_graph.items(), key=lambda item: item[1]))

    # Step 2: Get the first two and last two elements
    first_two_elements = list(sorted_cost_graph_by_value.items())[:2]
    last_two_elements = list(sorted_cost_graph_by_value.items())[-2:]

    # # Step 3: Find two consecutive elements with a non-zero difference
    # items_sorted = list(sorted_cost_graph_by_value.items())
    # min_difference = None
    #
    # for i in range(len(items_sorted) - 1):
    #     diff = abs(items_sorted[i][1] - items_sorted[i + 1][1])
    #     if diff != 0 and (min_difference is None or diff < min_difference[0]):
    #         min_difference = (diff, items_sorted[i], items_sorted[i + 1])

    # Step 3: Find two consecutive elements with the smallest difference
    items_sorted = list(sorted_cost_graph_by_value.items())
    differences = [(items_sorted[i], items_sorted[i + 1]) for i in range(len(items_sorted) - 1)]
    min_difference = min(differences, key=lambda x: abs(x[0][1] - x[1][1]))

    print("First two elements:", dict(first_two_elements))
    print("Last two elements:", dict(last_two_elements))
    print("Two consecutive elements with the smallest difference:", dict(min_difference))
    print("Difference: ", min_difference[0][1] - min_difference[1][1])

    return dict(first_two_elements), dict(last_two_elements), dict(min_difference)

