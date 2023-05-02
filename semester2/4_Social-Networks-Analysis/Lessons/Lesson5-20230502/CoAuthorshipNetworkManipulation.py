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
from DegreeCentralityDistribution import DegreeCentralityDistribution
from ReportTopNAuthors import ReportTopNAuthors

EXPECTED_USERS_NUM = 1155
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


def plot_graph(G: nx.classes.Graph, max_largest_components: int = 64, name: str = "Graph", with_labels: bool = False, block: bool = False, font_size: int = 6):
    plt.figure(1)
    plt.suptitle(f"{name}")
    nx.drawing.draw_networkx(G, with_labels=with_labels,
                             node_size=30, font_size=font_size)
    figure_file = os.path.join(OUTPUT_DIR, f"{name}_plot.png")
    plt.savefig(figure_file)

    largest_components = sorted(
        nx.connected_components(G), key=len, reverse=True)[:max_largest_components]

    plt.figure(2)
    plt.suptitle(
        f"{name}: {max_largest_components} largest Connected Components")
    n = len(largest_components)
    root = math.ceil(math.sqrt(len(largest_components)))
    for i, component in enumerate(largest_components):
        plt.subplot(root, root, i+1)
        H = G.subgraph(component)
        nx.drawing.draw_networkx(H, with_labels=with_labels,
                                 node_size=10, font_size=font_size)

    figure_file = os.path.join(
        OUTPUT_DIR, f"{name}_connected_components_plot.png")
    plt.savefig(figure_file)
    plt.show(block=block)


def main():
    print("start main")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    authors_df, icmb_dfs_dict = load_datafiles()
    W = create_W(icmb_dfs_dict)
    assert W.shape == (EXPECTED_USERS_NUM, EXPECTED_USERS_NUM)

    Wo = create_Wo(W)
    assert Wo.shape == (EXPECTED_USERS_NUM, EXPECTED_USERS_NUM)
    print('Wo\n', Wo)

    G = nx.classes.Graph(Wo)
    plot_graph(G)

    Degrees: np.ndarray = np.sum(Wo, axis=1)
    assert Degrees.shape == (EXPECTED_USERS_NUM, )

    H = DegreeCentralityDistribution(Degrees, OUTPUT_DIR)

    # Report top 10 authors according to Degree Centraluity.
    N = 10
    MeasureName = 'Degree Centrality'
    MeasureValues = Degrees
    ReportTopNAuthors(MeasureValues, MeasureName, N, authors_df)

    # block script so python doesn't exit
    plt.show()
    print("end main")


if __name__ == "__main__":
    main()
