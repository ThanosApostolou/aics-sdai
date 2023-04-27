import math
import typing
import numpy as np
import numpy.linalg as linalg
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.io
import os
import os.path
import pandas as pd
import networkx as nx
import networkx.drawing
import networkx.generators
import networkx.utils
import networkx.algorithms

DATAFILES_DIR = "./DataFiles"
OUTPUT_DIR = "./output"
YEARS = {
    2002,
    2003,
    2004,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013
}


def print_df(df: pd.DataFrame, name: str = ''):
    print(name)
    print(df.head())
    print(df.describe())
    print()


def apply_mat_element(element):
    if np.isscalar(element):
        return element
    else:
        newElement = element[0]
        return apply_mat_element(newElement)


def df_from_mat_or_csv(file_name: str, var_key: str) -> pd.DataFrame:
    csv_file = os.path.join(DATAFILES_DIR, f"{file_name}.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, encoding='utf-8', index_col=0, header=0)
        return df
    else:
        mat_file = os.path.join(DATAFILES_DIR, f"{file_name}.mat")
        mat_data = scipy.io.loadmat(mat_file, mat_dtype=False)
        mat_var = mat_data.get(var_key)
        if mat_var is None:
            raise Exception(
                f"could not find mat_var={mat_var} in file {mat_file}")

        df = pd.DataFrame(mat_var)
        for column in df.columns:
            df[column] = df[column].apply(apply_mat_element)

        print(f"{file_name}, df.shape", df.shape)
        df.to_csv(csv_file, encoding='utf-8', index=True, header=True)
        return df


def load_datafiles() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    authors_df = df_from_mat_or_csv("authors", "authors")

    icmb_dfs_dict: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        icmb_year_df = df_from_mat_or_csv(f"ICMB-{year}", f"array_{year}")
        icmb_dfs_dict[year] = icmb_year_df

    return (authors_df, icmb_dfs_dict)


def create_W(icmb_dfs_dict: dict[int, pd.DataFrame]) -> np.ndarray:
    shape = icmb_dfs_dict[2002].shape
    W = np.zeros(shape)
    for _, icmb_df in icmb_dfs_dict.items():
        icmb_array = icmb_df.to_numpy()
        W = np.add(W, icmb_array)

    return W


def create_Wo(W: np.ndarray) -> np.ndarray:
    # Create binary array with zeros or ones
    Wo = np.where(W > 0, 1, 0)
    # set all diag elements to 0, since we don't care about of each user's publications
    diag_indices = np.diag_indices(W.shape[0])
    Wo[diag_indices] = 0
    return Wo


# def show_graph_with_labels(adjacency_matrix, mylabels):
#     rows, cols = np.where(adjacency_matrix == 1)
#     edges = zip(rows.tolist(), cols.tolist())
#     gr = nx.Graph()
#     all_rows = range(0, adjacency_matrix.shape[0])
#     for n in all_rows:
#         gr.add_node(n)
#     gr.add_edges_from(edges)
#     nx.draw(gr, node_size=900, labels=mylabels, with_labels=True)
#     plt.show()

def plot_graph(G: nx.classes.Graph):
    plt.figure(1)
    plt.suptitle("Graph")
    nx.drawing.draw_networkx(G, with_labels=True)
    figure_file = os.path.join(OUTPUT_DIR, "graph_plot.png")
    plt.savefig(figure_file)

    largest_components = sorted(
        nx.connected_components(G), key=len, reverse=True)[:64]

    plt.figure(2)
    plt.suptitle("Graph's 64 largest Connected Components")
    n = len(largest_components)
    root = math.ceil(math.sqrt(len(largest_components)))
    for i, component in enumerate(largest_components):
        plt.subplot(root, root, i+1)
        H = G.subgraph(component)
        nx.drawing.draw(H, node_size=10, font_size=8)

    figure_file = os.path.join(
        OUTPUT_DIR, f"graph_connected_components_plot.png")
    plt.savefig(figure_file)
    plt.show()


def main():
    print("start main")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    authors_df, icmb_dfs_dict = load_datafiles()
    W = create_W(icmb_dfs_dict)
    Wo = create_Wo(W)
    print('Wo\n', Wo)

    G = nx.classes.Graph(Wo)

    plot_graph(G)
    print("end main")


if __name__ == "__main__":
    main()
