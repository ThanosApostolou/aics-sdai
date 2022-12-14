# This script file presents a thorough example of an end-to-end machine 
# learning project. The ultimate purpose is to build a machine model that will 
# predict the media housing price according to California Housing Prices 
# dataset.

# Import all the required Python frameworks for Phase I.
import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
#                               VARIABLES DEFINITION:
# ----------------------------------------------------------------------------
# Set the download url.
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# Set the local housing directory path.
HOUSING_PATH = os.path.join("datasets","housing")
# Set the complete housing url.
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# Set the local figures directory path.
FIGURES_PATH = "figures"
# Set the names of the housing numerical attributes.
NUMERICAL_ATTRIBUTES = ["Longitude", "Latitude", "Median Age", "Total Rooms", 
              "Total Bedrooms", "Population", "Households", "Median Income", 
              "Median House Value"]

# ----------------------------------------------------------------------------
#                            FUNCTION DEFINITIONS:
# ----------------------------------------------------------------------------
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # Handle possible exceptions when trying to download the data file.
    try: 
        urllib.request.urlretrieve(housing_url, tgz_path)
    except urllib.error.URLError as e:
        print(e.reason)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def report_attributes(data):
    for col_name in data.columns:
        print("\nReporting value counts information on {}".format(col_name))
        data_col = data[col_name]
        data_col_series = data_col.value_counts()
        print(data_col_series)

def generate_numeric_histograms(data, figures_path=FIGURES_PATH, 
                                numerical_attributes=NUMERICAL_ATTRIBUTES):
    os.makedirs(figures_path, exist_ok=True)
    for index, col_name in enumerate(housing.select_dtypes('number').columns):
        print("Generating histogram figure for {}".format(col_name))
        col_series = data[col_name]
        title_str = "{} Probability Density:".format(numerical_attributes[index])
        plot_histogram(col_series, title_str)

def plot_histogram(series, title_str):
    n, bins, patches = plt.hist(series, bins='auto', color='red',
                                alpha=0.7, rwidth=0.75)
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
    plt.show()
# ----------------------------------------------------------------------------
#            MAIN PROGRAM: PHASE I [DATA COLLECTION & INVESTIGATION]
# ----------------------------------------------------------------------------

# Download housing data.
fetch_housing_data()

# Load housing data.
housing = load_housing_data()

# Get and print the head data of the dataframe.
housing_head = housing.head()
print(housing_head)

# Get information about the housing dataframe.
housing.info()

# Find out what categories exist in the ocean_proximity attruibute and 
#  how many districts belong to each category.
ocean_proximity_series = housing["ocean_proximity"].value_counts()
print(ocean_proximity_series)

# Get and print a description for the numerical values of the dataframe.
housing_description = housing.describe()
print(housing_description)

# Create a histogram plot for each numerical attribute of the dataframe.
housing.hist(bins=50, figsize=(20,15))
plt.show()

# Generate and save histogram plots for each attribute in the dataframe.
generate_numeric_histograms(housing)