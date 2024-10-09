Features of the Script:
Dataset: Uses the built-in Iris dataset from sklearn for classification, which can be replaced with any dataset.
Model: A Random Forest classifier is used to train and make predictions.
Metrics: It calculates accuracy and provides a detailed classification report (precision, recall, f1-score) as well as a confusion matrix.
Visualization: Plots the confusion matrix and the feature importances to help understand model performance.
Model Persistence: Saves the trained model as a .pkl file using joblib for future use.

Install Dependencies:
pip install numpy pandas scikit-learn matplotlib seaborn joblib
