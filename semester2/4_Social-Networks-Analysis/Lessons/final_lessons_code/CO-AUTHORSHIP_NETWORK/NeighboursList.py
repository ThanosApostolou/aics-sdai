import numpy as np

def NeighboursList(Wo: np.ndarray) -> dict[int, list[int]]:
    """This function extracts the neighbours' list for corresponding to the 
    weight matrix W which is assummed to binary matrix indicating the 
    presence or absence of an edge between a given pair of nodes. Diagonal
    elements of matrix W are also assummed to be zero.

    NL is a cell array of vectors such that the element NL{u} stores the indices of nodes that are reachable from node u.
    """
    nodes_num = len(Wo)
    NL = {i: [] for i in range(nodes_num)}
    for v in range(nodes_num):
        vNeighborsTuple = np.where(Wo[v] == 1)
        vNeighbors = vNeighborsTuple[0]
        NL[v] = vNeighbors.tolist()

    return NL