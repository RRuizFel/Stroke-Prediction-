# Healthcare Stroke Data Analysis and Modeling

This markdown file contains information about the code for analyzing and modeling healthcare data related to stroke. The code covers various stages of data analysis, including data cleaning, visualization, feature engineering, statistical tests, and machine learning modeling using different algorithms such as Random Forest, XGBoost, and K-Nearest Neighbors.

## Contents

- **Data Loading and Preprocessing:** Load the data from a CSV file and clean the data by handling missing values and outliers.
- **Data Visualization and Exploratory Data Analysis:** Provide visualizations such as histograms, bar plots, and box plots to understand the distribution and relationships between variables.
- **Feature Engineering and Selection:** Perform feature engineering, including transforming categorical data and scaling numerical data. Select important features based on statistical tests.
- **Statistical Tests:** Use statistical tests such as Chi-Square and ANOVA to identify the significance of different features in relation to the target variable (stroke).
- **Machine Learning Modeling:** Train different machine learning models such as Random Forest, XGBoost, and K-Nearest Neighbors to predict the likelihood of stroke based on the input features.
- **Evaluation and Comparison of Models:** Evaluate the performance of each model using metrics such as accuracy, precision, recall, F1 score, and ROC AUC. Compare the performance of different models to choose the best one.

## Dataset

The dataset used in this code is the **Healthcare Stroke Dataset**. It contains information about various patients, including their demographic data and medical history, and whether or not they have experienced a stroke.

Key features of the dataset include:

- `id`: Unique identifier for each patient.
- `gender`: Gender of the patient (Male, Female, or Other).
- `age`: Age of the patient.
- `hypertension`: Whether the patient has hypertension (0 = No, 1 = Yes).
- `heart_disease`: Whether the patient has heart disease (0 = No, 1 = Yes).
- `ever_married`: Whether the patient has ever been married (Yes or No).
- `work_type`: Type of work the patient does (Private, Self-employed, etc.).
- `Residence_type`: Type of residence (Urban or Rural).
- `avg_glucose_level`: Average glucose level of the patient.
- `bmi`: Body mass index (BMI) of the patient.
- `smoking_status`: Smoking status of the patient (formerly smoked, never smoked, etc.).
- `stroke`: Target variable, indicating whether the patient has had a stroke (1) or not (0).

## Requirements

To run the code, ensure you have the following Python packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `statsmodels`
- `scipy`
- `scikit-learn`
- `xgboost`

You can install them using pip:

```shell
pip install numpy pandas matplotlib seaborn plotly statsmodels scipy scikit-learn xgboost
```
## Overview 

The code consists of the following sections:

- `Data Loading and Preprocessing`: The data is loaded from the CSV file and preprocessed, including handling missing values.
- `Data Visualization and Exploratory Data Analysis`: The code contains visualizations for various aspects of the data, such as age distribution, BMI, glucose levels, and categorical feature distributions.
- `Feature Engineering and Selection`: This section performs data transformations and feature engineering, including scaling numerical features and one-hot encoding categorical features.
- `Statistical Tests`: Chi-Square and ANOVA tests are used to assess feature importance in relation to the target variable.
- `Machine Learning Modeling`: Various models (Random Forest, XGBoost, K-Nearest Neighbors) are trained and evaluated.
- `Evaluation and Comparison`: Model performance is evaluated using metrics such as precision, accuracy, recall, and F1 score. ROC curves and AUC scores are also compared for each model.

## Outputs

The code produces the following visuzliations and outputs: 

- Data visualizations such as bar plots, histograms, and scatter plots.
- Statistical test results for feature importance.
- Performance metrics and reports for each machine learning model.
- ROC curves comparing the performance of different models.

