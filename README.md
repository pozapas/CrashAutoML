# CrashAutoML
## Crash Severity Analysis using Automated Machine Learning (AutoML) method

The code provided is for a Streamlit app named "Crash Severity AutoML." This app is designed to analyze crash severity data using automated machine learning (AutoML) methods. It allows users to upload CSV files containing crash data, select features and target variables for modeling, and handle data imbalances using SMOTE (Synthetic Minority Over-sampling Technique).

![Methodology](https://github.com/pozapas/CrashAutoML/blob/main/Methodology.svg)

Key features of the app include:

- **Data Upload and Preview**: Users can upload their crash data in CSV format and preview it in the app.
- **Feature Selection**: Users can select which features from their data to include in the model.
- **Target Variable Selection**: Users can choose the target variable for prediction.
- **Imbalance Handling Option**: An option to apply SMOTE for handling imbalanced datasets.
- **Automated Model Comparison**: The app runs a comparison of different machine learning models using PyCaret and displays the results, including F1 scores and other metrics.
- **Model Visualization**: The app generates various visualizations such as feature importance, confusion matrix, AUC-ROC curve, precision-recall curve, error analysis, class report, learning curve, and validation curve for the selected best model.
  
The repository will serve as a resource for users interested in crash severity analysis using AutoML and a showcase for the capabilities of Streamlit in creating interactive data science applications.
