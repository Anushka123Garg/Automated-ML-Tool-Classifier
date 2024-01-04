import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def normalize_data(X,y):

    # Step 1: Normalization/Standardization
    print("Choose a normalization/standardization technique:\n")
    print("1. Standard Scaler")
    print("2. Min-Max Scaler")
    print("3. Robust Scaler\n")
    normalization_option = int(input("Enter your choice: "))

    if normalization_option == 1:
        scaler = StandardScaler()
    elif normalization_option == 2:
        scaler = MinMaxScaler()
    elif normalization_option == 3:
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid normalization option")

    X_normalized = scaler.fit_transform(X)
    normalized = pd.DataFrame(X_normalized, columns= X.columns)

    return normalized, y 