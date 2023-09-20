# -------------------Imports---------------------------
import os
import sys
import numpy as np
import pandas as pd

from statistics import median
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

from smote import SMOTE
from sampling import generate_samples, delete_samples
from evaluate import measure_new_eod, measure_new_aod
from constant import column_labels

sys.path.append(os.path.abspath('../..'))


# Custom Exceptions
class EmptyList(Exception):
    pass


# Dataset Used Needs to be Large Enough to Have Data for all 27 Subsets

###======================Part 1: Code and Preprocessing Begins======================
base_path = str(sys.path[0])

input_file = base_path + '/Data/raw_state_NV.csv'
input_file_1 = base_path + '/Data/HMDA_2020_Data.csv'
input_file_2 = base_path + '/Data/HMDA_2019_Data.csv'
input_file_3 = base_path + '/Data/HMDA_2018_Data.csv'

final_file = base_path + '/Data/All_HMDA_Debiased.csv'
other_file = base_path + '/Data/newDatasetOrig.csv'

result_file = base_path + '/Results/ALL_HMDA_results.csv'

df_2020 = pd.read_csv(input_file_1, dtype=object).sample(n=755000)
df_2019 = pd.read_csv(input_file_2, dtype=object).sample(n=755000)
df_2018 = pd.read_csv(input_file_3, dtype=object).sample(n=755000)
dataset_orig = pd.concat([df_2020, df_2019, df_2018])
dataset_orig = dataset_orig.sample(frac=1)
dataset_orig.reset_index(drop=True, inplace=True)

action_taken_col = dataset_orig.pop('action_taken')
dataset_orig.insert(len(dataset_orig.columns), 'action_taken', action_taken_col)


#####-----------------------Declaring Cutting Functions-----------------
def removeExempt(array_columns, df):
    for startIndex in range(len(array_columns)):
        currentIndexName = df[df[array_columns[startIndex]] == "Exempt"].index
        df.drop(currentIndexName, inplace=True)


def removeBlank(array_columns, df):
    for startIndex in range(len(array_columns)):
        currentIndexName = df[df[array_columns[startIndex]] == ""].index
        df.drop(currentIndexName, inplace=True)


def bucketingColumns(column, arrayOfUniqueVals, nicheVar):
    currentCol = column
    for firstIndex in range(len(arrayOfUniqueVals)):
        try:
            dataset_orig.loc[(nicheVar == arrayOfUniqueVals[firstIndex]), currentCol] = firstIndex
        except:
            print("This number didn't work:\n", firstIndex)


#####------------------Scaling------------------------------------
def scale_dataset(processed_df):
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(processed_df), columns=processed_df.columns)
    return scaled_df


###------------------Preprocessing Function (includes Scaling)------------------------
def preprocessing(dataset_orig):
    # if you want 'derived_loan_product_type' column add here
    dataset_orig = dataset_orig[column_labels]

    # Below we are taking out rows in the dataset with values we do not care for. This is from lines 23 - 99.
    ###--------------------Sex------------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_sex'] == 'Male') |
                                (dataset_orig['derived_sex'] == 'Female') |
                                (dataset_orig['derived_sex'] == 'Joint')]
    dataset_orig['derived_sex'] = dataset_orig['derived_sex'].replace(['Female', 'Male', 'Joint'],
                                                                      [0, 1, 2])

    ###-------------------Races-----------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_race'] == 'White') |
                                (dataset_orig['derived_race'] == 'Black or African American') |
                                (dataset_orig['derived_race'] == 'Joint')]
    dataset_orig['derived_race'] = dataset_orig['derived_race'].replace(['Black or African American', 'White', 'Joint'],
                                                                        [0, 1, 2])

    ####----------------Ethnicity-------------------
    dataset_orig = dataset_orig[(dataset_orig['derived_ethnicity'] == 'Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Not Hispanic or Latino') |
                                (dataset_orig['derived_ethnicity'] == 'Joint')]
    dataset_orig['derived_ethnicity'] = dataset_orig['derived_ethnicity'].replace(
        ['Hispanic or Latino', 'Not Hispanic or Latino', 'Joint'],
        [0, 1, 2])
    # ----------------Action_Taken-----------------
    dataset_orig = dataset_orig[(dataset_orig['action_taken'] == '1') |
                                (dataset_orig['action_taken'] == '2') |
                                (dataset_orig['action_taken'] == '3')]

    dataset_orig['action_taken'] = dataset_orig['action_taken'].replace(['1', '2', '3'],
                                                                        [1, 0, 0])
    ######----------------Loan Product-------------------
    # assigns each unique categorical value a unique integer id
    dataset_orig['derived_loan_product_type'] = dataset_orig['derived_loan_product_type'].astype('category').cat.codes

    ####---------------Scale Dataset---------------
    dataset_orig = dataset_orig.apply(pd.to_numeric)
    dataset_orig = dataset_orig.dropna()
    dataset_orig = scale_dataset(dataset_orig)

    ####---------------Reset Indexes----------------
    dataset_orig.reset_index(drop=True, inplace=True)

    return dataset_orig


###---------Call Preprocessing to Create Processed_Scaled_DF---------------
processed_scaled_df = preprocessing(dataset_orig)
processed_scaled_shape = processed_scaled_df.shape

##------------------Check Initial Measures----------------------

processed_scaled_df["derived_sex"] = pd.to_numeric(processed_scaled_df.derived_sex, errors='coerce')
processed_scaled_df["derived_race"] = pd.to_numeric(processed_scaled_df.derived_race, errors='coerce')
processed_scaled_df["derived_ethnicity"] = pd.to_numeric(processed_scaled_df.derived_ethnicity, errors='coerce')
processed_scaled_df["action_taken"] = pd.to_numeric(processed_scaled_df.action_taken, errors='coerce')

np.random.seed(0)

# Divide into Train Set, Validation Set, Test Set
processed_scaled_train, processed_and_scaled_test = train_test_split(processed_scaled_df, test_size=0.2, random_state=0,
                                                                     shuffle=True)
X_train, y_train = processed_scaled_train.loc[:, processed_scaled_train.columns != 'action_taken'], \
                   processed_scaled_train['action_taken']
X_test, y_test = processed_and_scaled_test.loc[:, processed_and_scaled_test.columns != 'action_taken'], \
                 processed_and_scaled_test['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
clf.fit(X_train, y_train)
##-----------------------------------------------------------------------------------------------------------------------------------------------------


###===============Part 2: Working w/ Processed_Scaled_Df=================
ind_cols = ['derived_ethnicity', 'derived_race', 'derived_sex']
dep_col = 'action_taken'


def get_unique_df(processed_scaled_df, ind_cols):
    uniques = [processed_scaled_df[i].unique().tolist() for i in ind_cols]
    unique_df = pd.DataFrame(product(*uniques), columns=ind_cols)
    return unique_df


def split_dataset(processed_scaled_df, ind_cols):
    unique_df = get_unique_df(processed_scaled_df, ind_cols)
    combination_df = [pd.merge(processed_scaled_df, unique_df.iloc[[i]], on=ind_cols, how='inner') for i in
                      range(unique_df.shape[0])]
    return combination_df


global_unique_df = get_unique_df(processed_scaled_df, ind_cols)
print(global_unique_df)
combination_df = split_dataset(processed_scaled_df, ind_cols)


# -------------------------------Peformance Metrics--------------


# Calculates Recall Metric
def calculate_recall(TP, FP, FN, TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return recall


# Calculates Far Metric
def calculate_far(TP, FP, FN, TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return far


# Calculates Precision Metric
def calculate_precision(TP, FP, FN, TN):
    if (TP + FP) != 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    return prec


# Calculates Accuracy Metric
def calculate_accuracy(TP, FP, FN, TN):
    return (TP + TN) / (TP + TN + FP + FN)


# Calculates F1 Score
def calculate_F1(TP, FP, FN, TN):
    precision = calculate_precision(TP, FP, FN, TN)
    recall = calculate_recall(TP, FP, FN, TN)
    F1 = (2 * precision * recall) / (precision + recall)
    return round(F1, 2)


def evaluate_eod(df, biased_col):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], \
                       df_train['action_taken']
    X_test, y_test = df_test.loc[:, df_test.columns != 'action_taken'], \
                     df_test['action_taken']

    clf_1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_1.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return measure_new_eod(df_test, biased_col, y_pred)


def evaluate_aod(df, biased_col):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], \
                       df_train['action_taken']
    X_test, y_test = df_test.loc[:, df_test.columns != 'action_taken'], \
                     df_test['action_taken']

    clf_1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_1.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return measure_new_aod(df_test, biased_col, y_pred)


# ---------------------------------------- NOVEL FAIRNESS METRIC ----------------------------------
def evaluate_awi(df):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True)
    X_train, y_train = df_train.loc[:, df_train.columns != 'action_taken'], \
                       df_train['action_taken']
    X_test, y_test = df_test.loc[:, df_test.columns != 'action_taken'], \
                     df_test['action_taken']

    clf_1 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_1.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    acc = calculate_accuracy(TP, FP, FN, TN)
    recall = calculate_recall(TP, FP, FN, TN)
    precision = calculate_precision(TP, FP, FN, TN)
    far = calculate_far(TP, FP, FN, TN)
    F1 = calculate_F1(TP, FP, FN, TN)

    clf_2 = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
    clf_2.fit(X_train, y_train)
    removal_list = []

    count = 0
    for index, row in df_test.iterrows():

        pred_list = []
        row_ = [row.values[:len(row.values) - 1]]
        print("This is row", count, "of a total dataset of", df_test.shape[0])
        print('IM Global', global_unique_df)

        for _, other_row in global_unique_df.iterrows():
            current_comb = other_row.values[
                           :len(other_row.values)]  ## indexes are 2, 3, 4 for ethnicity, race, sex respectively
            original_ethnic, original_race, original_sex = row_[0][2], row_[0][3], row_[0][4]
            row_[0][2] = current_comb[0]
            row_[0][3] = current_comb[1]
            row_[0][4] = current_comb[2]
            y_current_pred = clf_2.predict(row_)[0]
            pred_list.append(y_current_pred)
            row_[0][2], row_[0][3], row_[0][4] = original_ethnic, original_race, original_sex

        print('Pred_list:', pred_list)
        num_unique_vals = len(set(pred_list))

        if num_unique_vals > 1:
            removal_list.append(index)
        elif num_unique_vals == 0:
            raise EmptyList
        count = count + 1

    removal_list = set(removal_list)
    total_biased_points = len(removal_list)
    print('total_biased_points:', total_biased_points)
    total_dataset_points = df_test.shape[0]

    # percentage of points unfairly predicted by the model
    AWI = total_biased_points / total_dataset_points

    return AWI, acc, precision, recall, far, F1


# AWI_initial, acc_intial, precision_intial, recall_intial, far_intial, F1_intial = evaluate_awi(processed_scaled_df)
EOD_sex_intial = evaluate_eod(processed_scaled_df, "derived_sex")
EOD_race_intial = evaluate_eod(processed_scaled_df, "derived_race")
EOD_ethnicity_intial = evaluate_eod(processed_scaled_df, "derived_ethnicity")
AOD_sex_intial = evaluate_aod(processed_scaled_df, "derived_sex")
AOD_race_intial = evaluate_aod(processed_scaled_df, "derived_race")
AOD_ethnicity_intial = evaluate_aod(processed_scaled_df, "derived_ethnicity")


def get_median_val(combination_df):
    current_total = 0
    array_of_bars = []
    for df in combination_df:
        pos_count = len(df[(df['action_taken'] == 1)])
        neg_count = len(df[(df['action_taken'] == 0)])
        current_total += (pos_count + neg_count)
        array_of_bars.append(pos_count)
        array_of_bars.append(neg_count)
    median_val = median(array_of_bars)
    return round(median_val)


mean_val = get_median_val(combination_df)

def RUS_balance(dataset_orig):
    if dataset_orig.empty:
        return dataset_orig
    # print('imbalanced data:\n', dataset_orig['action_taken'].value_counts())
    action_df = dataset_orig['action_taken'].value_counts()
    maj_label = action_df.index[0]
    min_label = action_df.index[-1]
    if maj_label == min_label:
        return dataset_orig
    df_majority = dataset_orig[dataset_orig.action_taken == maj_label]
    df_minority = dataset_orig[dataset_orig.action_taken == min_label]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority.index),  # to match minority class
                                       random_state=123)
    # Combine minority class with down sampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    df_downsampled.reset_index(drop=True, inplace=True)

    return df_downsampled


# TODO: simplify function
def smote_balance(comb_df):
    def apply_smote(df):
        df.reset_index(drop=True, inplace=True)
        cols = df.columns
        smt = SMOTE(df)
        df = smt.run()
        df.columns = cols
        return df

    X_train, y_train = comb_df.loc[:, comb_df.columns != 'action_taken'], comb_df['action_taken']

    train_df = X_train
    train_df['action_taken'] = y_train
    train_df["action_taken"] = y_train.astype("category")

    train_df = apply_smote(train_df)

    return train_df


def apply_balancing(combination_df, mean_val):
    smoted_list = []
    RUS_list = []
    for c in combination_df:
        pos_count = len(c[(c['action_taken'] == 1)])
        neg_count = len(c[(c['action_taken'] == 0)])

        current_max, current_min = max(pos_count, neg_count), min(pos_count, neg_count)

        if current_max < mean_val:
            print('current_max', current_max)
            print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
            smoted_df = smote_balance(c)
            smoted_list.append(smoted_df)
        elif (current_max > mean_val) and (current_min < mean_val):
            print('current_max2', current_max, current_min)
            print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
            diff_to_max = current_max - mean_val
            diff_to_min = mean_val - current_min
            if diff_to_max < diff_to_min:
                smoted_df = smote_balance(c)
                RUS_list.append(smoted_df)
            else:
                RUS_df = RUS_balance(c)
                smoted_list.append(RUS_df)
        elif (current_max > mean_val) and (current_min > mean_val):
            print('current_max3', current_max, current_min)
            print(c[['derived_ethnicity', 'derived_race', 'derived_sex', 'action_taken']])
            RUS_df = RUS_balance(c)
            RUS_list.append(RUS_df)

    super_balanced_RUS = []
    for df in RUS_list:
        num_decrease_of_0 = len(df[(df['action_taken'] == 0)]) - mean_val
        num_decrease_of_1 = len(df[(df['action_taken'] == 1)]) - mean_val

        print('Before Distribution\n', df['action_taken'].value_counts())

        df = delete_samples(num_decrease_of_0, df, 0)
        df = delete_samples(num_decrease_of_1, df, 1)

        print('After Distribution\n', df['action_taken'].value_counts())
        super_balanced_RUS.append(df)

    super_balanced_smote = []

    for df in smoted_list:
        num_increase_of_0 = mean_val - len(df[(df['action_taken'] == 0)])
        num_increase_of_1 = mean_val - len(df[(df['action_taken'] == 1)])

        print("Num of Increase:", num_increase_of_0, num_increase_of_1)
        print('Before Distribution\n', df['action_taken'].value_counts())

        df_zeros, df_ones = generate_samples(num_increase_of_0, num_increase_of_1, df, 'HMDA')
        df_added = pd.concat([df_zeros, df_ones])
        concat_df = pd.concat([df, df_added])
        concat_df = concat_df.sample(frac=1).reset_index(drop=True)
        print('After Distribution\n', concat_df['action_taken'].value_counts())
        super_balanced_smote.append(concat_df)

    def concat_and_shuffle(smote_version, RUS_version):
        concat_smote_df = pd.concat(smote_version)
        concat_RUS_df = pd.concat(RUS_version)
        total_concat_df = pd.concat([concat_RUS_df, concat_smote_df])
        total_concat_df = total_concat_df.sample(frac=1).reset_index(drop=True)

        print('Shuffle:', total_concat_df.head(50))
        return total_concat_df

    return concat_and_shuffle(super_balanced_smote, super_balanced_RUS)


new_dataset_orig = apply_balancing(combination_df, mean_val)
new_dataset_orig.to_csv(other_file, index=False)

# -----------------Situation Testing-------------------
X_train, y_train = new_dataset_orig.loc[:, new_dataset_orig.columns != 'action_taken'], new_dataset_orig['action_taken']

clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
clf.fit(X_train, y_train)

removal_list = []
# Generates list of rows to be removed
for index, row in new_dataset_orig.iterrows():
    pred_list = []
    row_ = [row.values[:len(row.values) - 1]]
    for other_index, other_row in global_unique_df.iterrows():
        current_comb = other_row.values[
                       :len(other_row.values)]  # indexes are 2, 3, 4 for ethnicity, race, sex respectively
        print('current_comb', current_comb)
        original_ethnic, original_race, original_sex = row_[0][2], row_[0][3], row_[0][4]
        row_[0][2] = current_comb[0]
        row_[0][3] = current_comb[1]
        row_[0][4] = current_comb[2]
        y_current_pred = clf.predict(row_)[0]
        pred_list.append(y_current_pred)
        print('pred_list', pred_list)
        row_[0][2], row_[0][3], row_[0][4] = original_ethnic, original_race, original_sex

    num_unique_vals = len(set(pred_list))
    if num_unique_vals > 1:
        removal_list.append(index)
    elif num_unique_vals == 0:
        raise EmptyList

removal_list = set(removal_list)
df_removed = pd.DataFrame(columns=new_dataset_orig.columns)

for index, row in new_dataset_orig.iterrows():
    if index in removal_list:
        df_removed = df_removed.append(row, ignore_index=True)
        balanced_and_situation_df = new_dataset_orig.drop(index)
##--------------------------Get Final Measures----------------------------

balanced_and_situation_df.to_csv(final_file)

# AWI_final, acc_final, precision_final, recall_final, far_final, F1_final = evaluate_awi(balanced_and_situation_df)
EOD_sex_final = evaluate_eod(balanced_and_situation_df, "derived_sex")
EOD_race_final = evaluate_eod(balanced_and_situation_df, "derived_race")
EOD_ethnicity_final = evaluate_eod(balanced_and_situation_df, "derived_ethnicity")
AOD_sex_final = evaluate_aod(processed_scaled_df, "derived_sex")
AOD_race_final = evaluate_aod(processed_scaled_df, "derived_race")
AOD_ethnicity_final = evaluate_aod(processed_scaled_df, "derived_ethnicity")
