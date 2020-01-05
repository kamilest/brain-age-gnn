
def add_population_graph_noise(graph, p, noise_amplitude):
    """Adds white Gaussian noise to the nodes of the population graph.

    :param graph: path to the population graph file.
    :param p: probability of adding noise.
    :param noise_amplitude: the variance of white noise.
    :return: the modified graph with increased noise.
    """

    pass


def remove_population_graph_edges(graph, p):
    """Removes graph edges with probability p.

    :param graph: path to the population graph file.
    :param p: probability of removing the edge.
    :return: the modified graph with fewer edges.
    """

    pass


def add_population_graph_edge_errors(graph, p):
    """Changes the graph connectivity by adding or removing edges.

    :param graph: path to the population graph file.
    :param p: probability of error (adding or removing an edge).
    :return: the modified graph with edge errors.
    """

    pass


def decrease_population_graph_train_set(graph, test_set_sizes):
    """Decreases the training set (more unlabeled nodes).

    :param graph: path to the population graph file.
    :param test_set_sizes:
    :return:
    """


def measure_predictive_power_drop():
    """Measures the drop in performance metrics with increased noise or more missing data.

    :return: the range of values at different modification levels.
    """
    pass

