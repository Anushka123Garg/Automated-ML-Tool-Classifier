# Project Title
*
Development of an automated ML-based tool, a comprehensive Python script for classification problems that will run on any given input data matrix.
Input matrix Format: Standard Feature x Instance matrix (.csv) file.
Label_column should be at the last of the matrix.

## Table of Contents
**
1. Installing Dependencies
2. Getting Started
3. Data Loading
4. Data Normalization
5. Feature selection
6. Cross-validation
7. Model training and evaluation
8. Visualizations

### Installing Dependencies
***
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. seaborn

Install the dependencies using pip: pip install {dependencies_name}

### Getting Started
***
1. IDE Used - VS Code 
2. Install Python and dependencies
3. Run the main script that contains all other '.py' files: 'Main.py'
    All the files should be in same folder
4. Test using different datasets. I have tested on four datasets : breast_cancer.csv, Iris.csv, faults.csv and children anemia.csv      dataset
5. A text file is generated that contains all the results and a pdf file is generated that contains all the plots and graphs.

### Data Loading
***
The 'Load.py' module is responsible for loading the dataset. The loaded data is then printed, and the problem type (binary or multiclass) is determined.

### Data Normalization
***
The 'Normalization.py' module normalizes the loaded data using three normalization techniques: Standard Normalization, Min-Max scaler normalization, Robust Scaler Normalization. The normalized data is then printed for reference.

### Feature Selection
***
The 'Feature_selection.py' module is used for feature selection. 5 techniques used are: SelectKBest with chi-squared, SelectKBest with ANOVA F-statistic, Recursive Feature Elimination (RFE), Mutual Information", Variance Threshold. The selected features are stored for further processing.

### Cross Validation
***
The 'Cross.py' module is utilized for performing Stratified K-fold cross validation, K-fold cross-validation and LeaveOneOut. Cross-validation scores and mean accuracy are printed.

### Model Training and Evaluation
***
The Model.py script uses one of the five models: Random Forest Classifier, Logistic Regression, KNN, Support Vector Classifier, Decision Tree model. The model is trained using a train-test split, and accuracy on the test set is printed. Cross-validation scores and a classification report and blind dataset predictability are also provided.

#### Visualizations
****

1. Pairplot
2. Correlation heatmap
3. Confusion matrix
4. ROC Curve (for binary classification) or ROC Curve for Multiclass