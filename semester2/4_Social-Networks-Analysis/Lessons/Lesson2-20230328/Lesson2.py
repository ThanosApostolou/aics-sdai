import math
import numpy as np
import numpy.linalg as linalg
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=900, labels=mylabels, with_labels=True)
    plt.show()


def main():
    print("start main")
    W = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    print(W)

    W_2 = linalg.matrix_power(W, 2)
    print('W_2', W_2)

    print('factorial_100', math.factorial(100))

    # show_graph_with_labels(W, ['A', 'B', 'C'])

    G = nx.classes.Graph(W, labels=['a', 'b', 'c'])
    nx.draw(G, with_labels=True)
    plt.show()
    print("end main")


if __name__ == "__main__":
    main()