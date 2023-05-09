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

# Import all the required Python frameworks for Phase II.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Import all the required Python frameworks for Phase III
from pandas.plotting import scatter_matrix
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

# ----------------------------------------------------------------------------
#            MAIN PROGRAM: PHASE II [DATA PARTITIONING]
# ----------------------------------------------------------------------------

# Create an additional income category attribute with five categories.
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1,2,3,4,5])
plot_histogram(housing["income_cat"], "Income Catetory Probability Density Distribution")

# Perform a random partitioning of the dataset into training and testing subsets.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# Performe a stratified partitioning of the dataset into training and testing 
# subsets based on the income_cat attribute.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    print("================================================")
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    print("================================================")
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
# Generate 3 additional series to count the frequency of each income category
# within the original dataset, the stratified test dataset and the randomly 
# sampled test dataset.
overall_series = housing["income_cat"].value_counts() / len(housing)
overall_series = overall_series.rename('overall')
stratified_series = strat_test_set["income_cat"].value_counts() / len(strat_test_set)
stratified_series = stratified_series.rename('stratified')
random_series = test_set["income_cat"].value_counts() / len(test_set)
random_series = random_series.rename('random')
# Concatenate the three series into a new dataframe.
income_df = pd.concat([overall_series, stratified_series, random_series], axis=1)

# Remove the income_cat attribute so the data is back to its original state.
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# ----------------------------------------------------------------------------
#            MAIN PROGRAM: PHASE IΙI [DATA VISUALIZATION]
# ----------------------------------------------------------------------------

# Create a copy of the training dataset.
housing = strat_train_set.copy()
# Plot the training data.
# s: marker size (can be scalar or array of size equal to size of x or y)
# c: color of sequence of colors for markers
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,8),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
# Save figure.
figure_url = os.path.join(FIGURES_PATH, "HousingHeatMap.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# Compute the correlation matrix between every pair of attributes.
corr_matrix = housing.corr() 
# Compute the degree of correlation between each attribute and the median 
# house value.
median_house_value_corr = corr_matrix["median_house_value"].sort_values(
                                                            ascending=False)

# Define a promising set of house attributes and generate their pairwise
# scatter plots.
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# Save figure.
figure_url = os.path.join(FIGURES_PATH, "AttributesCorrelation.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# The most promising house attribute turns out to be the median income. 
# Visualize the scatter plot between median house value and median income.
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
# Save figure.
figure_url = os.path.join(FIGURES_PATH, "IncomePriceCorrelation.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# Consider the following combinations of the original housing attributes.
housing ["rooms_per_household"] = housing ["total_rooms"]/housing ["households" ] 
housing ["bedrooms_per_room"] = housing [ "total_bedrooms" ]/housing ["total_rooms"] 
housing ["population_per_household"] = housing ["population"]/housing ["households"] 

# Compute the correlation matrix between every pair of attributes.
corr_matrix_new = housing.corr() 
# Compute the degree of correlation between each attribute and the median 
# house value.
new_median_house_value_corr = corr_matrix_new["median_house_value"].sort_values(
                                                            ascending=False)