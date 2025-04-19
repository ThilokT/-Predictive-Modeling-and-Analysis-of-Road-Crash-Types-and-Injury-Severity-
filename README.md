# Crash-Type Prediction: EDA and Predictive Modeling

## Purpose
This notebook explores traffic crash data to predict the type of crash based on various features. The primary objectives are:
- Conducting Exploratory Data Analysis (EDA) to understand the underlying patterns.
- Preprocessing data for model training.
- Training machine learning models to predict crash types and evaluating their performance.

---

## Tools Used

### Data Analysis & Visualization:
- **pandas**: For data manipulation, cleaning, and handling.
- **numpy**: For numerical operations and array manipulations.
- **matplotlib**: For basic data visualization (e.g., histograms, box plots).
- **seaborn**: For advanced visualizations (e.g., heatmaps, distribution plots).
  
### Machine Learning:
- **scikit-learn**: For implementing and training machine learning models.
  - **LabelEncoder**: To encode categorical variables.
  - **train_test_split**: To split data into training and testing sets.
  - **classification models**: Various classifiers are tested, such as Decision Trees, Random Forest, and Logistic Regression.
  - **evaluation metrics**: Accuracy, confusion matrix, classification report.

---

## How Data is Analyzed

### Data Preprocessing:
- **Handling Missing Data**: Any missing or null values are identified and appropriately handled (imputation or removal).
- **Outlier Detection**: Box plots and other visual methods are used to detect and address outliers that may skew the model performance.
- **Categorical and Numerical Feature Grouping**: Categorical variables are grouped and encoded for model training, while numerical variables are analyzed for distribution and potential transformations (e.g., normalization).
  
### Exploratory Data Analysis (EDA):
- **Categorical Variable Distribution**: Bar plots are created for categorical variables to visualize their frequency and distribution.
- **Categorical vs Target Variable**: Heatmaps show how categorical variables relate to the target (crash type).
- **Numerical Data Distribution**: Histograms and KDE plots display the distribution of continuous features, helping to identify skewness and the need for data transformation.
- **Outlier Analysis**: Box plots highlight any outliers in numerical features that could impact model performance.
- **Correlation Analysis**: Pair plots and correlation matrices help identify redundant features and relationships between them.

---

## Models Trained

### 1. **Random Forest**:
   - An ensemble method using multiple decision trees to improve accuracy and reduce overfitting by averaging results across trees.
   
### 2. **K-Nearest Neighbors (KNN)**:
   - A non-parametric method used for classification by comparing feature similarity with the nearest neighbors.
   
### 3. **Logistic Regression**:
   - A simple yet effective model for binary classification, useful for understanding the impact of each feature on the crash type.

---

## Models Evaluation

After training and evaluating the models, the **Random Forest** and **Logistic Regression** models both performed well, achieving an accuracy of **0.79**. The **K-Nearest Neighbors (KNN)** model achieved a slightly lower accuracy of **0.77**.
