import numpy as np
from numpy import int32


def enqueue(Q: list[int], element: int) -> list[int]:
    """This is a subfunction implementing the enqueue operation within a 
    queue which is realzed as a vector of elements
    """

    Q.append(element)
    return Q


def dequeue(Q: list[int]) -> tuple[list[int], int]:
    """This is a subfunction implementing the dequeue operation within a 
    queue queue which is realized as a vector of elements.    
    """

    element = Q[0]
    Q = Q[1:]

    return Q, element


def ConnectedComponents(NL: dict[int, list[int]]) -> dict[int, list[int]]:
    """This function extracts the connnected components of a given undirected 
    graph whose neighbours' list NL is given as input. C is a cell array of 
    vectors so that each vector stores the indices of each connected component.
    """

    # Initialize the dictionary C storing the connected components of the graph.
    C: dict[int, list[int]] = {}
    # Get the number of graph nodes.
    nodes_num = len(NL)
    # Mark all nodes as unvisited.
    visited = np.full(nodes_num, False, dtype=bool)
    # Initialize the number of connected components found so far.
    components_num = 0
    for v in range(nodes_num):
        # If v is not visited yet, it's the start of a newly discovered
        # component containing v.
        if not visited[v]: # Process the component containing v.
            component: list[int] = [] # Initialize component container.
            Q: list[int] = [] # Initialize queue for implementing breadth-first search.
            Q = enqueue(Q, v) # Start the traversal from node v.
            visited[v] = True
            while len(Q) is not 0:
                Q, w = dequeue(Q) # w is a node in this component.
                component.append(w)
                # Get all nodes neighbouring w.
                node_neighbours = NL[w]
                # Traverse each unvisited node neighbouring w.
                for node_index in range(len(node_neighbours)):
                    node = node_neighbours[node_index]
                    if not visited[node]:
                        # Another node within the current component has been found.
                        visited[node] = True
                        Q = enqueue(Q, node)
            
            C[components_num] = component
            components_num = components_num + 1

    return C
