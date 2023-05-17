import numpy as np
import pandas as pd


def ReportTopNAuthors(MeasureValues: np.ndarray, MeasureName: str, N: int, authors_df: pd.DataFrame):
    """This function reports the top N authors ranked by the measure identified
    by the input parameter MeasureName. The corresponding measure values are
    stored within the vector MeasureValues. The number of N and the complete
    list of authors names are also given as input to the function.
    """

    SortedIndices = np.flip(np.argsort(MeasureValues))
    SortedValues = MeasureValues[SortedIndices]
    TopNSortedValues = SortedValues[:N]
    TopNSortedIndices = SortedIndices[:N]

    first_names = authors_df["2"].to_numpy(dtype='str')
    TopNAuthorsFirstNames = first_names[TopNSortedIndices]
    sur_names = authors_df["1"].to_numpy(dtype='str')
    TopNAuthorsSurNames = sur_names[TopNSortedIndices]
    #  Report Top N Authors' List.
    print(f"Top {N} Authors according to {MeasureName}\n")
    for k in range(0, N):
        print(
            f"{TopNAuthorsSurNames[k]} {TopNAuthorsFirstNames[k]}: {TopNSortedValues[k]}")
