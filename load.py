import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data():
    file_path = input("Enter the file path of the CSV file: ")
    # Load data from CSV file
    df = pd.read_csv(file_path)

    label_col = df.columns[-1]
    print(label_col)

    if 'ID' in df.columns:
        X = df.drop(['ID', label_col], axis=1)
    else:
        X = df.drop(label_col, axis=1)

    y_encoded = df[label_col]

    imputer = SimpleImputer(strategy='most_frequent')

    missing_values = X.columns[X.isnull().any()].tolist()
    if missing_values:
        X[missing_values] = imputer.fit_transform(X[missing_values])

    if y_encoded.dtype == 'O':
        label_encoder_y = LabelEncoder()
        y_encoded = label_encoder_y.fit_transform(y_encoded)

    for column in X.columns:
        if X[column].dtype == 'O' and not X[column].apply(lambda x: isinstance(x, (int, float))).all():
            label_encoder_X = LabelEncoder()
            X[column] = label_encoder_X.fit_transform(X[column])

    return X, pd.Series(y_encoded, name=label_col)

