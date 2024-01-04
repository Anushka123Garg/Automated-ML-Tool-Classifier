from load import load_data
from normalization import normalize_data
from feature_selection import select_features
from cross import cross_validate
from models import choose_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import sys
import builtins 

with open('results.txt', 'w') as file:
    
    def custom_print(*args, **kwargs):
        file.write(' '.join(map(str, args)) + '\n')
        builtins.print(*args, **kwargs)

    print = custom_print

    # Load data
    X, y_encoded = load_data()
    print(X)

    no_of_classes = len(y_encoded.unique())
    problem_type = "binary" if no_of_classes == 2 else "multiclass"
    print(f"This is a {problem_type} classification problem.")

    # Normalize data
    normalized_X, y_encoded = normalize_data(X, y_encoded)
    print("\nNormalized Data:")
    print(normalized_X)

    # Select features
    selected_features = select_features(normalized_X, y_encoded)

    # Cross-validation
    cv = cross_validate(selected_features, y_encoded)

    # Model
    model = choose_model(problem_type)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y_encoded, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Blind dataset
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on Blind Dataset: {test_accuracy:.2f}")

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, selected_features, y_encoded, cv=cv, scoring='accuracy')
    print("\nCross-Validation Scores:", scores)

    print("\nMean Accuracy:", scores.mean())

    with np.errstate(divide='warn', invalid='warn'):
        classification_report_str = classification_report(y_test, y_pred, zero_division=1)
        print(classification_report_str)    

    print = builtins.print

with PdfPages('output10.pdf') as pdf:

    # Pairplot  
    sns.pairplot(selected_features.join(pd.Series(y_encoded, name='target')), hue='target')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Correlation heatmap
    corr_matrix = selected_features.join(pd.Series(y_encoded, name='target')).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    pdf.savefig()
    plt.close()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    pdf.savefig()
    plt.close()

    # ROC Curve for Binary Classification
    if problem_type == "binary":
        if hasattr(model, 'predict_proba'):
            y_binary = label_binarize(y_test, classes=np.unique(y_test))  
            y_score = model.predict_proba(X_test)

            fpr, tpr, _ = roc_curve(y_binary, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            pdf.savefig()
            plt.close()
        else:
            print("Warning: predict_proba is not available for the chosen model. Skipping ROC curve for binary classification.")

    # ROC Curve for Multiclass Classification
    elif problem_type == "multiclass":

        if hasattr(model, 'predict_proba'):
            classifier = OneVsRestClassifier(model)
            try:
                y_score_multiclass = classifier.fit(X_train, y_train).predict_proba(X_test)
            except AttributeError:
                # Handle the case where predict_proba is not available
                print("Warning: predict_proba is not available for the chosen model.")
                y_score_multiclass = None
        else:
            print("Warning: predict_proba is not available for the chosen model.")
            y_score_multiclass = None

        if y_score_multiclass is not None:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            classifier = OneVsRestClassifier(model)
            y_score_multiclass = classifier.fit(X_train, y_train).predict_proba(X_test)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(len(np.unique(y_test))):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score_multiclass[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC curves
            plt.figure(figsize=(10, 6))
            colors = ['blue', 'orange', 'green']  

            for i, color in zip(range(len(np.unique(y_test))), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = {:.2f}) for class {}'.format(roc_auc[i], i))

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
            plt.legend(loc='lower right')
            pdf.savefig()
            plt.close()

