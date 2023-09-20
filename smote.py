import random
from collections import Counter
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    def __init__(self, data, neighbor=5, r=2, up_to_num=None):
        """
        Initialize SMOTE instance.

        :param data: DataFrame - the last column must be class label
        :param neighbor: int - number of nearest neighbors to select
        :param r: int - distance metric
        :param up_to_num: int - size of minorities to over-sample
        """
        self.data = data
        self.neighbor = neighbor
        self.r = r
        self.up_to_num = up_to_num or self.get_majority_num()

    def get_majority_num(self):
        """Get the number of majority class instances."""
        label_counts = Counter(self.data.iloc[:, -1].values)
        return max(label_counts.values())

    def _get_neighbor(self, data_no_label):
        """Get a random neighbor of a random sample from the data."""
        rand_sample_idx = random.randint(0, len(data_no_label) - 1)
        rand_sample = data_no_label[rand_sample_idx]

        knn = NearestNeighbors(n_neighbors=self.neighbor, p=self.r).fit(data_no_label)
        _, neighbors = knn.kneighbors(rand_sample.reshape(1, -1))
        rand_neighbor = data_no_label[random.choice(neighbors[0])]

        return rand_neighbor, rand_sample

    def run(self):
        """Perform SMOTE."""
        total_data = self.data.values.tolist()
        label_counts = Counter(self.data.iloc[:, -1].values)
        majority_num = max(label_counts.values())

        for label, num in label_counts.items():
            if num < majority_num:
                to_add = self.up_to_num - num

                data_with_label = self.data[self.data.iloc[:, -1] == label]
                data_no_label = data_with_label.iloc[:, :-1].values

                for _ in range(to_add):
                    rand_neighbor, sample = self._get_neighbor(data_no_label)
                    new_row = [max(0, sample[i] + (sample[i] - one) * random.random()) for i, one in
                               enumerate(rand_neighbor)]
                    new_row.append(label)
                    total_data.append(new_row)

        return pd.DataFrame(total_data)
