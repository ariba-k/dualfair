import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from constant import column_labels

def get_neighbors(df, knn):
    """Fetch three samples: a random parent and its two nearest neighbors."""

    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    neighbors_indices = knn.kneighbors(parent_candidate.values.reshape(1, -1), 3, return_distance=False)[0]

    return parent_candidate, df.iloc[neighbors_indices[1]], df.iloc[neighbors_indices[2]]


def generate_samples(no_of_samples_zeros, no_of_samples_ones, df, df_name):
    """
    Generate synthetic samples using KNN.

    :param no_of_samples_zeros: Number of samples with class 0.
    :param no_of_samples_ones: Number of samples with class 1.
    :param df: DataFrame to generate samples from.
    :param df_name: Name of the dataframe, specific handling for 'HMDA'.
    :return: DataFrames for class 0 and 1 samples.
    """

    total_data_zero, total_data_one = [], []
    knn = NearestNeighbors(n_neighbors=5).fit(df)
    count_zero, count_one = 0, 0

    while count_one < no_of_samples_ones or count_zero < no_of_samples_zeros:
        cr, f = 0.8, 0.8
        parent, child1, child2 = get_neighbors(df, knn)
        new_sample = []

        for key, value in parent.items():
            if isinstance(value, bool):
                new_sample.append(value if cr < random.random() else not value)
            elif isinstance(value, str):
                new_sample.append(random.choice([value, child1[key], child2[key]]))
            elif isinstance(value, list):
                temp_list = [item if cr < random.random() else int(item + f * (child1[key][idx] - child2[key][idx]))
                             for idx, item in enumerate(value)]
                new_sample.append(temp_list)
            else:
                new_sample.append(abs(value + f * (child1[key] - child2[key])))

        if new_sample[-1] == 0 and count_zero < no_of_samples_zeros:
            total_data_zero.append(new_sample)
            count_zero += 1
        elif new_sample[-1] == 1 and count_one < no_of_samples_ones:
            total_data_one.append(new_sample)
            count_one += 1

    final_df_zero, final_df_one = pd.DataFrame(total_data_zero), pd.DataFrame(total_data_one)

    # Rename columns if the dataframe is 'HMDA'
    if df_name == 'HMDA':
        column_array = column_labels
        rename_dict = {c: column_array[c] for c in range(len(column_array))}
        final_df_zero.rename(columns=rename_dict, errors="raise", inplace=True)
        final_df_one.rename(columns=rename_dict, errors="raise", inplace=True)

    return final_df_zero, final_df_one

def delete_samples(df: pd.DataFrame, target_column: str, target_value: int,
                          num_samples_to_delete: int) -> pd.DataFrame:
    """
    Delete a specified number of samples from a dataframe based on a target value in a specified column.

    :param df: Input dataframe
    :param target_column: Column to check for target value
    :param target_value: Value to check for deletion
    :param num_samples_to_delete: Number of samples to delete
    :return: Updated dataframe
    """

    eligible_rows = df[df[target_column] == target_value].index.tolist()

    if len(eligible_rows) < num_samples_to_delete:
        raise ValueError("Not enough eligible rows to delete")

    rows_to_delete = random.sample(eligible_rows, num_samples_to_delete)
    df = df.drop(rows_to_delete)
    df.reset_index(drop=True, inplace=True)

    return df
