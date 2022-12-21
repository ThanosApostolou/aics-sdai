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

# Import all the required Python frameworks for Phase IV.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Import all thee required Python frameworks for phase V.
from classes.ratio_attributes_adder import RatioAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# Import all the required Python frameworks for phase VI.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# ----------------------------------------------------------------------------
#                               VARIABLES DEFINITION:
# ----------------------------------------------------------------------------
# Set the download url.
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# Set the local housing directory path.
HOUSING_PATH = os.path.join("datasets", "housing")
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


def generate_numeric_histograms(housing, figures_path=FIGURES_PATH,
                                numerical_attributes=NUMERICAL_ATTRIBUTES):
    os.makedirs(figures_path, exist_ok=True)
    for index, col_name in enumerate(housing.select_dtypes('number').columns):
        print("Generating histogram figure for {}".format(col_name))
        col_series = housing[col_name]
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


def main():
    # ----------------------------------------------------------------------------
    #            MAIN PROGRAM: PHASE I [DATA COLLECTION & INVESTIGATION]
    # ----------------------------------------------------------------------------

    # Download housing data.
    fetch_housing_data()

    # Load housing dat
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
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Generate and save histogram plots for each attribute in the dataframe.
    generate_numeric_histograms(housing)

    # ----------------------------------------------------------------------------
    #            MAIN PROGRAM: PHASE II [DATA PARTITIONING]
    # ----------------------------------------------------------------------------

    # Create an additional income category attribute with five categories.
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
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
                 s=housing["population"] / 100, label="population", figsize=(10, 8),
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
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    # Compute the correlation matrix between every pair of attributes.
    corr_matrix_new = housing.corr()
    # Compute the degree of correlation between each attribute and the median
    # house value.
    new_median_house_value_corr = corr_matrix_new["median_house_value"].sort_values(
        ascending=False)

    # ----------------------------------------------------------------------------
    #            MAIN PROGRAM: PHASE IV [PREPARE DATA FOR MACHINE LEARNING PART A
    # ----------------------------------------------------------------------------

    # Revert to a clean training set by assigning the housing dataframe variable
    # with a copy of the stratified training subset of data. The copy operation is
    # conducted by dropping a series of the original training dataframe that will
    # eventually used as the target regression variable (label). Mind that the
    # dropping operation does not affect the initial training dataframe.
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # ----------------------------------------------------------------------------
    # Data Cleaning: Replace missing values in each series of numerical values
    #                within the dataframe with the corresponding median value.
    # ----------------------------------------------------------------------------

    # Instantiate the SimpleImputer class.
    imputer = SimpleImputer(strategy="median")
    # Create a copy of the reverted housing dataset by dropping its non-numerical
    # values.
    housing_numeric = housing.drop("ocean_proximity", axis=1)
    # Fit the imputer instance to the training numeric data.
    imputer.fit(housing_numeric)

    # The imputer has simply computed the median value for each attribute and stored
    # the result in its statistics_ instance variable.
    imputer_statistics = imputer.statistics_
    median_values = housing_numeric.median().values
    print("Imputer Statistics = {}".format(imputer_statistics))
    print("Median Values = {}".format(median_values))

    # Use the trained imputer to transform to transform the training set by
    # replacing the missing values with the learned medians.
    X = imputer.transform(housing_numeric)
    # The result is a plain NumPy array which can be transformed back to a pandas
    # dataframe.
    housing_numeric_transformed = pd.DataFrame(X, columns=housing_numeric.columns,
                                               index=housing_numeric.index)

    # ----------------------------------------------------------------------------
    # Handling Categorical Attributes: Replace categorical attributes with the
    #                                  corresponding one-hot encoding.
    # ----------------------------------------------------------------------------

    # Create a copy of the categorical attributes of the dataset. In this way, the
    # resulting variable is a dataframe and not series.
    housing_categoric = housing[["ocean_proximity"]]
    # Instantiate the catetorical one-hot encoder class.
    one_hot_encoder = OneHotEncoder()
    # Fit the one-hot encoder to the training categoric data and obtain the
    # corresponding sparse vector respresentation.
    housing_categoric_1hot_sparse = one_hot_encoder.fit_transform(housing_categoric)
    # Get the corresponding full matrix version.
    Y = housing_categoric_1hot_sparse.toarray()
    # Print the original text categories learned by the one-hot encoder.
    print("Original Categories = \n{}".format(one_hot_encoder.categories_))
    # The result is once again a plain NumPy array which can be trasformed back to
    # a pandas dataframe. This operation, however, needs to exclude the original
    # column name since there exist five columns in total within the new dataframe.
    # Thus, we  need to create a new list of strings with the new column names.
    categoric_names = ["ocean_proximity_{}".format(s)
                       for s in range(0, Y.shape[1])]
    housing_categoric_encoded = pd.DataFrame(Y, columns=categoric_names,
                                             index=housing_categoric.index)
    # Acquire the complete version of the dataframe by combining the numeric and the
    # categorical data.
    housing_complete = pd.concat([housing_numeric_transformed,
                                  housing_categoric_encoded], axis=1)

    # ----------------------------------------------------------------------------
    #            MAIN PROGRAM: PHASE V [PREPARE DATA FOR MACHINE LEARNING PART B]
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # Generating Addtional Attributes: Incorporate additional numeric attributes
    #                                  by utilizing the custom transformer.
    # ----------------------------------------------------------------------------

    # Set the indices of the original features that will be used to generate the
    # additional ratio-based features.
    rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6
    # Create a dictionary object to define the new feature names along with the
    # pairs of indices of the original features defining the new feature values as
    # their ratio.
    ratio_features = {"rooms_per_household": (rooms_idx, households_idx),
                      "population_per_housold": (population_idx, households_idx),
                      "bedrooms_per_room": (bedrooms_idx, rooms_idx)}

    # Extract the feature_pairs list from the previously defined dictionary object.
    feature_pairs = list(ratio_features.values())

    # Instantiate the custom transformer class in order to generate the extened
    # version of the housing dataset.
    attribute_adder = RatioAttributesAdder(feature_pairs)

    # Utilize the instanciated custom transformer to acquire the extended numpy
    # array of housing features.
    Z = attribute_adder.transform(housing.values)

    # Create a new list storing the names of the extended housing features dataframe.
    extended_names = list(housing) + list(ratio_features)

    # Acquire the extened version of the housing dataset.
    # Mind that this version of the extened dataframe contains missing values since
    # the ratio attribute adder was called on an instance of the housing dataset
    # where the imputer has not been called yet.
    housing_extended = pd.DataFrame(Z, columns=extended_names, index=housing.index)

    # ----------------------------------------------------------------------------
    # Data Transformation Pipelines: Unify the previously defined manipulation
    #                                processes on both the numeric and categorical
    #                                attributes by defining appropriate data
    #                                transformation pipelines. Apply the
    #                                aforementioned transformations on both training
    #                                and testing subsets of the dataset.
    # ----------------------------------------------------------------------------

    # Acquire clean versions of the training and test sets along with the coresponding
    # target regression variables.
    housing_train_features = strat_train_set.drop("median_house_value", axis=1)
    housing_train_labels = strat_train_set["median_house_value"].copy()
    housing_test_features = strat_test_set.drop("median_house_value", axis=1)
    housing_test_labels = strat_test_set["median_house_value"].copy()

    # Define the data transformation pipeline for the numerical features.
    numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                                 ('attribs_adder', RatioAttributesAdder(feature_pairs)),
                                 ('min_max_scaler', MinMaxScaler()), ])

    # Define the data transformation pipeline for the target variables.
    labels_pipeline = Pipeline([('min_max_scaler', MinMaxScaler()), ])

    # Initialize list objects storing the names of the numerical and categorical
    # attributes.
    numeric_attributes = list(housing_numeric)
    categoric_attributes = ["ocean_proximity"]

    # Define the full data transformation pipeline for both numeric and categoric
    # features.
    features_pipeline = ColumnTransformer([("numeric", numeric_pipeline, numeric_attributes),
                                           ("categoric", OneHotEncoder(), categoric_attributes), ])

    # Get the Xtrain and Ytrain variables for the underlying regression problem by
    # applying the previously defined data transformations on the training-related
    # variables. Mind that the targe regression variables have to be reshaped into
    # a column vector.
    Xtrain = features_pipeline.fit_transform(housing_train_features)
    Ytrain = labels_pipeline.fit_transform(housing_train_labels.values.reshape(-1, 1))

    # Get the Xtest and Ytest variables for the underlying regression problem by
    # applying the previously defined data transformations on the testing-related
    # variables. Mind that the targe regression variables have to be reshaped into
    # a column vector.
    Xtest = features_pipeline.fit_transform(housing_test_features)
    Ytest = labels_pipeline.fit_transform(housing_test_labels.values.reshape(-1, 1))

    # ----------------------------------------------------------------------------
    #            MAIN PROGRAM: PHASE VI [NEURAL NETWORK-BASED REGRESSION]
    # ----------------------------------------------------------------------------

    # Get the dimensionality of the input space.
    dimensionality = Xtrain.shape[1]
    # Get the number of training samples.
    records_num = Xtrain.shape[0]

    # Define the architecture of the neural network for the underlying regression
    # task.
    model = Sequential()
    model.add(Dense(4 * dimensionality, input_dim=dimensionality, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(3 * dimensionality, activation='sigmoid'))
    model.add(Dense(2 * dimensionality, activation='sigmoid'))
    model.add(Dense(dimensionality, activation='sigmoid'))
    model.add(Dense(int(dimensionality / 2), activation='sigmoid'))
    model.add(Dense(int(dimensionality / 4), activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    # Compile the neural network model.
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    # Train the neural network model.
    history = model.fit(Xtrain, Ytrain, epochs=400, batch_size=int(records_num / 8),
                        verbose=1, validation_split=0.2)

    # Get a list of the training metrics.
    metrics = history.history.keys()
    print(metrics)

    # Visualize training history for MSE.
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model Accuracy in terms of MSE')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    plt.show()

    # Visualize training history for MAE.
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model Accuracy in terms of MAE')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    plt.show()

    # Visualize training history for LOSS.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy in terms of LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    plt.show()

    # Get the target regression values for the testing subset of the dataset.
    Ytest_est = model.predict(Xtest)
    # Get the residual values between the actual and predicted values.
    Res = Ytest - Ytest_est

    # Generate a histogram plot for the acquired residual values.
    plt.hist(Res, density=True, bins=30)  # density=False would make counts
    plt.title('Error PDF')
    plt.ylabel('Probability')
    plt.xlabel('Error')
    plt.grid()
    plt.show()
    print()

# MAIN PROGRAM: PHASE II
# Create an additional income category attribute with five categories
if __name__ == '__main__':
    main()
