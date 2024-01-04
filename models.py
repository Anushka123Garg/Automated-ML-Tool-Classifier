from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def choose_model(problem_type):

    print("Choose a machine learning model:\n")
    print("1. Random Forest")
    print("2. Logistic Regression")
    print("3. K-Nearest Neighbors")
    print("4. Support Vector Classifier (SVC)")
    print("5. Decision Tree\n")
    model_option = int(input("Enter your choice: "))

    if model_option == 1:
        model = RandomForestClassifier(random_state=42)
    elif model_option == 2:
        if problem_type == "binary":
            model = LogisticRegression(random_state=42)
        else:
            model = LogisticRegression(multi_class='multinomial', random_state=42)
    elif model_option == 3:
        model = KNeighborsClassifier()
    elif model_option == 4:
        model = SVC(random_state=42)
    elif model_option == 5:
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Invalid model option")

    return model



