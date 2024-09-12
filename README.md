# Leveraging Machine Learning to Predict Liver Cirrhosis Outcomes from Clinical Data
## Overview
This project aims to leverage machine learning techniques to predict outcomes for patients with liver cirrhosis using clinical data. The key questions addressed by the project are:
* What are the most critical features in predicting liver cirrhosis outcomes?
* Can machine learning models accurately predict the probability of different outcomes, such as survival or death, for patients with cirrhosis?
* How can we handle class imbalance in the dataset, especially the minority class of transplant outcomes?

## Project Description
**Motivation** Liver cirrhosis represents a significant global health challenge, often leading to severe complications or death without timely intervention. Predicting patient outcomes accurately can greatly enhance clinical decision-making and improve patient care. This project explores machine learning models to assist in this prediction.

**Why this project?** Given the progressive nature of liver cirrhosis and the complexity of its clinical manifestations, there is a strong need for predictive tools that can help clinicians anticipate disease outcomes and tailor interventions accordingly. Machine learning, with its ability to handle large, complex datasets, is well suited for this task.

**What problem did it solve?** The project developed predictive models that can assist clinicians by predicting patient outcomes based on clinical features. These models provide actionable insights into which medical markers are significant for predicting cirrhosis outcomes and suggest future improvements for managing liver cirrhosis.

**What did we learn?** The most critical features for predicting liver cirrhosis outcomes are bilirubin and albumin levels. Machine learning models, especially Gradient Boosting and Random Forest, accurately predicted survival, death, and transplant outcomes. To address class imbalance, class weight adjustments outperformed SMOTE, with the Gradient Boosting model using class weights achieving the best performance, yielding a validation log loss of 0.459.

## Usage 
1. Install required Python packages:
    ```bash
    pip install pandas numpy matplotlib tensorflow scikit-learn
    ```

2. Navigate to the project directory:
    ```bash
    cd <path to repository>
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open the notebook in the Jupyter Lab interface.

## Technologies Used 
* **Pandas**: For loading, cleaning, and manipulating the clinical dataset.
* **Numpy**: Used for numerical operations and data manipulation.
* **Matplotlib/Seaborn**: For visualising data distributions and correlations between features.
* **Scikit-learn**: Applied for machine learning models, including Random Forest, Gradient Boosting, and data preprocessing tasks like scaling and imputation.
* **XGBoost**: A powerful gradient boosting algorithm used for classification tasks.
* **Imbalanced-learn (SMOTE)**: To handle class imbalance through synthetic oversampling of the minority class.

## Data Description
The dataset used for this project is synthetic, derived from the [Mayo Clinic Liver Cirrhosis Study (1974-84)](https://www.kaggle.com/competitions/playground-series-s3e26/). It includes 7,905 patients with 19 clinical features, such as age, bilirubin, albumin, cholesterol, and liver function tests. The outcome variable had three categories Alive with cirrhosis (C), Alive with liver transplant (CL), Dead (D).
