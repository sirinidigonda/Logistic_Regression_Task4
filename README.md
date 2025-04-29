Tools Used:
Python
Pandas
Scikit-learn
Matplotlib
NumPy

Dataset:
Breast Cancer Wisconsin (Diagnostic) Dataset

Features used:
radius_mean
texture_mean
perimeter_mean
area_mean
smoothness_mean
compactness_mean
concavity_mean
concave points_mean
symmetry_mean
fractal_dimension_mean
radius_se
texture_se
perimeter_se
area_se
smoothness_se
compactness_se
concavity_se
concave points_se
symmetry_se
fractal_dimension_se
radius_worst
texture_worst
perimeter_worst
area_worst
smoothness_worst
compactness_worst
concavity_worst
concave points_worst
symmetry_worst
fractal_dimension_worst

Target Variable:
Diagnosis (Malignant or Benign)

Steps Followed:
Imported and preprocessed the dataset.
Cleaned and handled missing data.
Split the dataset into features (X) and target (y).
Split the dataset into training and testing sets (80-20 split).

Built and trained models using:
Logistic Regression
Random Forest Classifier
Tuned the Random Forest model for better performance.

Evaluated the models using:
Accuracy
Confusion Matrix
Classification Report
Visualized the results by plotting confusion matrices and classification reports.

Model Performance:

Logistic Regression
Accuracy: 98.25%
Confusion Matrix:
42  1
1 70
Classification Report:
precision    recall  f1-score   support
    0       0.98      0.98      0.98        43
    1       0.99      0.99      0.99        71
    
Random Forest
Accuracy: 96.49%
Confusion Matrix:
40  3
1 70
Classification Report:
precision    recall  f1-score   support
    0       0.98      0.93      0.95        43
    1       0.96      0.99      0.97        71
    
Tuned Random Forest
Accuracy: 96.49%
Confusion Matrix:
40  3
1 70
Classification Report:
precision    recall  f1-score   support
    0       0.98      0.93      0.95        43
    1       0.96      0.99      0.97        71

Graph: Uploaded

Observations:
The Logistic Regression model achieved a high accuracy of 98.25%.
The Random Forest model also performed well, with an accuracy of 96.49%.
Both models showed a good balance between precision and recall for both classes.
The confusion matrix indicates that the models are able to predict most benign and malignant cases correctly.

Conclusion:
Successfully built and evaluated machine learning models (Logistic Regression and Random Forest) for predicting breast cancer diagnosis.
Learned how to interpret evaluation metrics such as accuracy, confusion matrix, and classification report, and visualized model performance.

