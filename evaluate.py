import numpy as np

def calculate_confusion_matrix_elements(test_df, biased_col, y_pred, bias_value):
    test_df = test_df.copy()
    current_pred_col = f'current_pred_{biased_col}'
    test_df[current_pred_col] = y_pred

    TP = np.where((test_df['action_taken'] == 1) &
                  (test_df[current_pred_col] == 1) &
                  (test_df[biased_col] == bias_value), 1, 0).sum()

    TN = np.where((test_df['action_taken'] == 0) &
                  (test_df[current_pred_col] == 0) &
                  (test_df[biased_col] == bias_value), 1, 0).sum()

    FN = np.where((test_df['action_taken'] == 1) &
                  (test_df[current_pred_col] == 0) &
                  (test_df[biased_col] == bias_value), 1, 0).sum()

    FP = np.where((test_df['action_taken'] == 0) &
                  (test_df[current_pred_col] == 1) &
                  (test_df[biased_col] == bias_value), 1, 0).sum()

    return TP, TN, FN, FP

def calculate_matrix_up_and_p(test_df, biased_col, y_pred):
    TP_p, TN_p, FN_p, FP_p = calculate_confusion_matrix_elements(test_df, biased_col, y_pred, 0.5)
    TP_up, TN_up, FN_up, FP_up = calculate_confusion_matrix_elements(test_df, biased_col, y_pred, 0)

    print(TP_p, TN_p, FN_p, FP_p, TP_up, TN_up, FN_up, FP_up)
    return TP_p, TN_p, FN_p, FP_p, TP_up, TN_up, FN_up, FP_up

def calculate_ratio(num, den):
    return num / den if den != 0 else 0

def calculate_equal_opportunity_difference(TP_p, TN_p, FN_p, FP_p, TP_up, TN_up, FN_up, FP_up):
    TPR_p = calculate_ratio(TP_p, TP_p + FN_p)
    TPR_up = calculate_ratio(TP_up, TP_up + FN_up)

    diff = TPR_p - TPR_up
    print(f"TPR_priv: {TPR_p}, TPR_unpriv: {TPR_up}, Difference: {diff}")
    return diff

def calculate_equalizied_odds_difference(TP_p, TN_p, FN_p, FP_p, TP_up, TN_up, FN_up, FP_up):
    TPR_p = calculate_ratio(TP_p, TP_p + FN_p)
    TPR_up = calculate_ratio(TP_up, TP_up + FN_up)
    FPR_p = calculate_ratio(FP_p, FP_p + TN_p)
    FPR_up = calculate_ratio(FP_up, FP_up + TN_up)

    diff = ((FPR_up - FPR_p) + (TPR_up - TPR_p)) * 0.5
    print(f"TPR_priv: {TPR_p}, TPR_unpriv: {TPR_up}, FPR_priv: {FPR_p}, FPR_unpriv: {FPR_up}, Difference: {diff}")
    return diff

def run_pipeline(test_df, biased_col, y_pred, metric_function):
    metrics = calculate_matrix_up_and_p(test_df, biased_col, y_pred)
    return metric_function(*metrics)

def measure_new_aod(test_df, biased_col, y_pred):
    return run_pipeline(test_df, biased_col, y_pred, calculate_equalizied_odds_difference)

def measure_new_eod(test_df, biased_col, y_pred):
    return run_pipeline(test_df, biased_col, y_pred, calculate_equal_opportunity_difference)

# Additional helper metric functions
def calculate_recall(TP, _, FN, __):
    return calculate_ratio(TP, TP + FN)

def calculate_far(_, FP, __, TN):
    return calculate_ratio(FP, FP + TN)

def calculate_precision(TP, FP, _, __):
    return calculate_ratio(TP, TP + FP)

def calculate_accuracy(TP, FP, FN, TN):
    return calculate_ratio(TP + TN, TP + TN + FP + FN)
