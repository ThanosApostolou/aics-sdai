import os
import tarfile
import urllib
import urllib.request
import urllib.error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# GLOBAL VARIABLES DEFINITION
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
FIGURES_PATH = "figures"

NUMERICAL_ATTRIBUTES = ["Longitude", "Latitude", "Median Age", "Total Rooms",
                        "Total Bedrooms", "Population", "Households", "Median Income",
                        "Median House Value"]


# FUNCTIONS DEFINITION SECTION
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    try:
        urllib.request.urlretrieve(housing_url, tgz_path)
    except urllib.error.URLError as e:
        print(e.reason)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def plot_histogram(series: pd.Series, title_str: str):
    n, bins, patches = plt.hist(series, bins='auto', color='red', alpha=0.7, rwidth=0.75)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    plt.title(title_str)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    filename = series.name + ".png"
    figure_url = os.path.join(FIGURES_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    # plt.show()


def generate_numeric_histograms(data: pd.DataFrame, figures_path=FIGURES_PATH,
                                numerical_attributes=NUMERICAL_ATTRIBUTES):
    os.makedirs(figures_path, exist_ok=True)
    for index, col_name in enumerate(data.select_dtypes("number").columns):
        print("Generating histogram figure for {}".format(col_name))
        col_series = data[col_name]
        title_str = "{} Probability Density:".format(numerical_attributes[index])
        plot_histogram(col_series, title_str)
    data.hist(bins=50, figsize=(20,15))
    plt.show()


def report_attributes(data: pd.DataFrame):
    for col_name in data.columns:
        print("\n Reporting value count information on {}".format(col_name))
        data_col = data[col_name]
        data_col_series = data_col.value_counts()
        print(data_col_series)


def test_housing_values(myhousing):
    print(myhousing.head())
    myhousing.info()
    print(myhousing.loc(0))


def main():
    print('PyCharm')
    # DOWNLOAD HOUSING DATASET
    fetch_housing_data()
    # LOAD HOUSING DATA
    housing = load_housing_data()
    # report_attributes(housing)
    generate_numeric_histograms(housing)
    print()
    # test_housing_values(housing)


if __name__ == '__main__':
    main()
