import math
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

DATAFILES_DIR = "./DataFiles"

def load_datafiles():
    authors_mat_file = os.path.join(DATAFILES_DIR, "authors.mat")
    authors_mat = scipy.io.loadmat(authors_mat_file)
    authors = authors_mat.get("authors")
    authors_array = pd.DataFrame(authors)
    print(authors_array)

def main():
    print("start main")
    load_datafiles()
    
    print("end main")


if __name__ == "__main__":
    main()