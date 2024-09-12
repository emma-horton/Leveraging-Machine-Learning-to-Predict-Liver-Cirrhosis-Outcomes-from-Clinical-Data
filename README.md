# Leveraging Machine Learning to Predict Liver Cirrhosis Outcomes from Clinical Data
This project leverages machine learning techniques to predict outcomes for liver cirrhosis patients using clinical data. We built predictive models, including Gradient Boosting and Random Forest classifiers, to assess key medical features and provide clinical insights. The models are evaluated using metrics like AUROC, F1-score, and LogLoss. Additionally, methods for handling imbalanced data, such as SMOTE and class weighting, were implemented. Our final models showed promising results in predicting patient outcomes and have potential for real-world clinical application.

## Purpose
This project was developed by a group of five students as part of our coursework for the ‘Knowledge Discovery and Data Mining’ module at the University of St Andrews. The primary aim was to apply machine learning techniques to predict liver cirrhosis outcomes using clinical data. Our goal was to build and evaluate predictive models that not only enhance our technical skills but also provide insights that can potentially aid clinicians in improving patient care and decision-making in real-world medical scenarios.

## Aims 
* To address the challenge of unbalanced data through techniques such as SMOTE or class weighting.
* To construct multiple machine learning models aimed at predicting multi-class outcomes based on clinical attributes.
* To evaluate the performance of the models using relevant metrics and analyses.
* To present findings in a clear and accessible format suitable for a non-technical audience, demonstrating the real-world impact of machine learning in healthcare.

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

