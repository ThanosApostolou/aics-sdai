import pandas as pd
import numpy as np

def ReportTopNConnectedComponents(C: dict[int, list[int]], N: int, authors_df: pd.DataFrame):
    """This function reports the top N (measured by size) connected components 
    of the co-authorship network that are stored in cell array C. 
    The number of top N components and the initial authors' list are passed 
    input arguments to the function.
    """

    # Get the number of connected components.
    components_num = len(C)
    # Get the size of each connected component.
    components_sizes = np.zeros(components_num, dtype=int)
    for k in range(components_num):
        components_sizes[k] = len(C[k])

    # Sort connected components sizes in descending order.
    SortedComponentsIndices = np.argsort(components_sizes)[::-1]
    SortedComponentsSizes = components_sizes[SortedComponentsIndices]
    # Get the top N connected components sizes and corresponding indices.
    TopNComponentsSizes = SortedComponentsSizes[:N]
    print('TopNComponentsSizes', TopNComponentsSizes)
    TopNComponentsIndices = SortedComponentsIndices[:N]

    # Report Connected Components.
    # Cycle through the top N connected components:
    for n in range(N):
        component_index = TopNComponentsIndices[n]
        component_size = TopNComponentsSizes[n]
        component = C[component_index]
        print(f"Component {component_index} of size {component_size}")
        # Cycle through the authors of each connected component:
        for m in range(component_size):
            author_index = component[m]
            author_firstname = authors_df['2'].iloc[author_index]
            author_lastname = authors_df['1'].iloc[author_index]
            print(f"{m}: {author_lastname} {author_firstname}")
