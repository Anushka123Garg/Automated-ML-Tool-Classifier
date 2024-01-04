from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def select_features(X_normalized, y):
    print("Shape of X_normalized before feature selection:\n", X_normalized.shape)

    print("Choose a feature selection technique:\n")

    print("1. SelectKBest with chi-squared only if you have used Min-max normalization")
    print("2. SelectKBest with ANOVA F-statistic")
    print("3. Recursive Feature Elimination (RFE)")
    print("4. Mutual Information")
    print("5. Variance Threshold\n")
    feature_selection_option = int(input("Enter your choice: "))

    if feature_selection_option == 1:            # requires (non negative data) min max normalization
        k = int(input("Enter the value of k for SelectKBest: "))
        selector = SelectKBest(chi2, k=k)
    # print(X_normalized)
    
    elif feature_selection_option == 2:
        k = int(input("Enter the value of k for SelectKBest: "))
        selector = SelectKBest(f_classif, k=k)

    elif feature_selection_option == 3:
        k = int(input("Enter the number of features to select: "))
        if k > X_normalized.shape[1]:
            raise ValueError("The specified number of features exceeds the total number of features.")
        selector = RFE(RandomForestClassifier(), n_features_to_select=k)

    elif feature_selection_option == 4:
        k = int(input("Enter the value of k for SelectKBest with mutual_info_classif: "))
        selector = SelectKBest(mutual_info_classif, k=k)

    elif feature_selection_option == 5:
        threshold_value = float(input("Enter the variance threshold value (low threshold: 0.1-0.01, medium threshold: 0.5-0.2, high threshold: 0.9-0.8, no threshold:0.0): "))
        selector = VarianceThreshold(threshold=threshold_value)
    else:
        raise ValueError("Invalid feature selection option")

    X_selected = selector.fit_transform(X_normalized, y)
    
    print("Shape of X_normalized before feature selection:", X_normalized.shape)
    print("Shape of X_selected after feature selection:", X_selected.shape)

    selected_features_df = pd.DataFrame(X_selected, columns=X_normalized.columns[:X_selected.shape[1]])

    return selected_features_df

