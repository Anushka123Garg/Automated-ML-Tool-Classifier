from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, LeaveOneOut

def cross_validate(X, y):
    
    no_of_classes = len(y.unique())
    problem_type = "binary" if no_of_classes == 2 else "multiclass"

    print("Choose a cross-validation technique:\n")
    print("1. Stratified K-Fold (Binary)" if problem_type == "binary" else "1. Stratified K-Fold (Multiclass)")
    print("2. K-Fold")
    print("3. Leave-One-Out")
    cross_val_option = int(input("Enter your choice: "))

    if cross_val_option == 1:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type == "binary" else KFold(n_splits=5, shuffle=True, random_state=42)
    elif cross_val_option == 2:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    elif cross_val_option == 3:
        cv = LeaveOneOut()
    else:
        raise ValueError("Invalid cross-validation option")
    
    return cv

