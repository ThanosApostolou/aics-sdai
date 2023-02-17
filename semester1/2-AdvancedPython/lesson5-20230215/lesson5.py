# IMDB Manipulation

# Import all required Python libraries

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

## Main Program Variables
# (0): dataset: np array of strings
# (1): dataframe: original dataset in its primal form
# (2): ratings_num_df: new dataframe storing the number of rated items per unique user
# (3): ratings_span_df: new dataframe storing the timespan in days for each user.
# (4): minimum_ratings and maximum_ratings
#      => ratings_df
# (5): final_df

# function from previous lesson: plot_histogram(series, title_str)
from plot_histogram import plot_histogram

# Global vars
FIGURES_PATH = "figures"


def main():
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Define the datafolder path
    datafolder = "datafiles"
    os.makedirs(datafolder, exist_ok=True)

    # set the file containing the dataset
    datafile = "./Dataset.npy"

    # Load the dataset in an np array
    dataset = np.load(datafile)

    # define splitter lambda function
    spliter = lambda s: s.split(",")
    # Apply the splitter lambda function on the string np array
    dataset = np.array([spliter(x) for x in dataset])

    # Set the pickle file for
    pickle_file = os.path.join(datafolder, "dataframe.pkl")

    # check the existence of the previously defined file
    dataframe: pd.DataFrame
    if os.path.exists(pickle_file):
        # Load the dataframe
        dataframe = pd.read_pickle(pickle_file)
    else:
        # create the dataframe for the first time
        dataframe = pd.DataFrame(dataset, columns=["user", "item", "rating", "date"])
        dataframe["user"] = dataframe["user"].apply(lambda s: np.int64(s.replace("ur", "")))
        dataframe["item"] = dataframe["item"].apply(lambda s: np.int64(s.replace("tt", "")))
        dataframe["rating"] = dataframe["rating"].apply(lambda s: np.int64(s))
        dataframe["date"] = dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe.to_pickle(pickle_file)

    # Get the unique users in the dataset
    users = dataframe["user"].unique()
    users_num: int = len(users)

    # Get the unique items in the dataset
    items = dataframe["item"].unique()
    items_num: int = len(items)

    # Get the toal number of ratings in the dataset
    ratings_num: int = dataframe["rating"].shape[0]

    print("INITIAL DATASET: {0} number of unique users and {1} number of unique items".format(users_num, items_num))
    print("INITIAL DATASET: {0} number of existing ratings".format(ratings_num))

    # define the pickle file storing the number of ratings per user
    pickle_file = os.path.join(datafolder, "ratings_num_df.pkl")
    ratings_num_df: pd.DataFrame
    if os.path.exists(pickle_file):
        ratings_num_df = pd.read_pickle(pickle_file)
    else:
        ratings_num_df = dataframe.groupby("user")["rating"].count().sort_values(ascending=False).reset_index(
            name="ratings_num")
        ratings_num_df.to_pickle(pickle_file)

    # define the pickle file storing the number of ratings per user
    pickle_file = os.path.join(datafolder, "ratings_span_df.pkl")
    ratings_span_df: pd.DataFrame
    if os.path.exists(pickle_file):
        ratings_span_df = pd.read_pickle(pickle_file)
    else:
        ratings_span_df = dataframe.groupby("user")["date"].apply(lambda date: max(date) - min(date)).sort_values(
            ascending=False).reset_index(name="ratings_span")
        ratings_span_df.to_pickle(pickle_file)

    # Join the ratings_num_df and ratings_span_df in a new dataframe
    ratings_df = ratings_num_df.join(ratings_span_df.set_index("user"), on="user")
    ratings_df["ratings_span"] = ratings_df["ratings_span"].dt.days

    # Set the threshold values for the minimum and maximum number of ratings per user.
    minimum_ratings = 100
    maximum_ratings = 300

    # Discard all users that fall outside of
    reduced_ratings_df = ratings_df.loc[
        (ratings_df["ratings_num"] >= minimum_ratings) & (ratings_df["ratings_num"] <= maximum_ratings)]

    # Call the histogram function
    plot_histogram(reduced_ratings_df["ratings_num"], "NUmber of Ratings per user")
    plot_histogram(reduced_ratings_df["ratings_span"], "Time Span of Ratings per user")

    # Discard all user that do not pers to the previous range of ratings
    final_df: pd.DataFrame = dataframe.loc[dataframe["user"].isin(reduced_ratings_df["user"])].reset_index()
    final_df = final_df.drop("index", axis=1)

    # Get the reduced number of users and ratings in the finals form of the dataset
    final_users = final_df["user"].unique()
    final_users_num = len(final_users)
    final_items = final_df["item"].unique()
    final_items_num = len(final_items)
    final_ratings_num = final_df.shape[0]

    print("Reduce Dataset: {0} number of reduced users and {1} number of reduced items".format(final_users_num,
                                                                                               final_items_num))
    print("Reduce Dataset: {0} number of reduced ratings".format(final_ratings_num))

    sorted_final_users = np.sort(final_users)
    sorted_final_items = np.sort(final_items)

    final_users_dict = dict(zip(sorted_final_users, list(range(0, final_users_num))))
    final_items_dict = dict(zip(sorted_final_items, list(range(0, final_items_num))))

    final_df["user"] = final_df["user"].map(final_users_dict)
    final_df["item"] = final_df["item"].map(final_items_dict)

    # Get a grouped version of the final dataframe based on the unique final users.
    users_group = final_df.groupby("user")

    W = np.zeros((final_users_num, final_users_num))
    CommonRatings = np.zeros((final_users_num, final_users_num))

    # INitialize a dictionary object that will store the set of items rated by any given user.
    user_items_dict = {}
    for user in final_users:
        user_index = final_users_dict[user]
        user_items = set(users_group.get_group(user_index)["item"])
        user_items_dict[user_index] = user_items

    user_ids = list(user_items_dict.Keys())
    user_ids.sort()
    user_items_dict = {user_index: user_items_dict[user_index] for user_index in user_ids}

    # Generate matrixes W and CommonRatings
    # We want W(source_user, target_user)
    # and CommonRatings(source_user, target_user)
    for source_user in user_items_dict:
        for target_user in user_items_dict:
            intersection_items = user_items_dict[source_user].intersection(user_items_dict[target_user])
            union_items = user_items_dict[source_user].union(user_items_dict[target_user])
            W[source_user, target_user] = len(intersection_items/len(union_items))
            CommonRatings[source_user, target_user] = len(intersection_items)
    print()


if __name__ == '__main__':
    main()
