# Lung cancer classification
This project by Keith Chen Yong, aimed at his first E2E personal project for Lung Cancer classification. <br>

## Project Title
Prediction of Lung Cancer Using Classification Algorithms

##  Introduction
This initiative aims to predict the severity of lung cancer diagnosis in individuals based on personal information, sourced from Kaggle. The significance of early cancer detection cannot be overstated, as it is crucial for the timely administration of treatments that can significantly enhance survival rates. The core task of this project involves the application of classification algorithms to facilitate this prediction. Initially, three algorithms are employed: *Logistic Regression*, *Random Forest Classification*, and *Support Vector Machines* (SVM). These were selected for their proven effectiveness in similar tasks, with the structure of the code designed to accommodate the introduction of additional algorithms or hyperparameter fine-tuning of existing ones for optimal performance.

## Installation
All the necessary dependencies are specified in requirements.txt.

## Usage
Upon pulling or manually triggering the workflow in GitHub, the necessary installations from requirements.txt and the execution of the run.sh bash script will be completed. This will be followed by the execution of the main.py script. This Python script sequentially triggers various Python files responsible for tasks such as data ingestion (from a .db file named cancer_patient_data_sets within a folder titled data), data preprocessing, model building, k-fold cross-validation, and lastly, evaluation of the algorithms. Users will be able to see the k-fold cross-validation results, offering a comprehensive view of the dataset's behavior under different models and various classification performance metrics for each algorithm. <br>
![image](https://github.com/KeithChenYong/aiap-chen-yong-427J/assets/133010489/c5b911b6-aa4b-409e-a6eb-ce90b37a0fde)

User will be able to see the k-fold cross-validation results, offering a holistic view of the dataset's behavior under different models and various classification Performance metrics for each algorithm.<br>
**Note** Given the project's focus on the early detection of lung cancer, the <b>recall</b> metric (also known as sensitivity) should be prioritized to minimize the risk of overlooking actual cases of lung cancer. This prioritization underscores the project's commitment to ensuring the highest possible accuracy in identifying true positives.
| Performance Metric (Classification)          | Definition |
|-----------------|------------|
| Confusion Matrix| - **True Positives (TP):** Correctly predicted positive.<br>- **True Negatives (TN):** Correctly predicted negative.<br>- **False Positives (FP):** Incorrectly predicted positive.<br>- **False Negatives (FN):** Incorrectly predicted negative. |
| Accuracy        | - The ratio of correctly predicted to the total observations.<br>- Formula: `(TP + TN) / (TP + TN + FP + FN)`. |
| Precision       | - The ratio of correctly predicted positive to the total predicted positive observations.<br>- Also known as Positive Predictive Value.<br>- Formula: `TP / (TP + FP)`. |
| Recall **Important**          | - The ratio of correctly predicted positive to all observations in actual class - yes.<br>- AKA Sensitivity or True Positive Rate.<br>- Formula: `TP / (TP + FN)`. |
| Specificity     | - The ratio of correctly predicted negative to all observations in actual class - no.<br>- AKA True Negative Rate.<br>- Formula: `TN / (TN + FP)`. |

## Configuration
Users have the flexibility to modify the following
1. Modify the k value in crossval.py for cross-validation
2. Fine-tuning hyperparameters in model.py to improve model performance or tailor it to different datasets
3. Introduce new algorithms in main.py for exploring additional analytical approaches. *Note: Additional of new algorithms will require modify of model.py*

| ML Algorithm            | Justification |
|-------------------------|---------------|
| Logistic Regression     | - Simple, fast, and efficient for linearly separable data.<br>- Good baseline model. |
| Random Forest Classifier | - Handles non-linear data well due to ensemble of decision trees.<br>- Robust to overfitting with large number of features.<br>- Less sensitive to class imbalance (provided data are not imbalance). |
| Support Vector Classification | - Effective in high-dimensional spaces.<br>- Kernel trick allows for non-linear classification.<br>- Soft margin approach enhances generalization. |

## Contributing
Contributions are welcome! Please submit bug reports or feature requests via the [GitHub issue tracker](https://github.com/KeithChenYong/aiap-chen-yong-427J/issues).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details. 

## Credits
**AISG** as the source of data provision

## Contact
Author Name:   Chen Yong
Author Email:  keith.chenyong@gmail.com
For questions or feedback, please contact [me](mailto:keith.chenyong@gmail.com).

## Additional Documentation
For guidance on hyperparameter optimization, please refer to the following link.
1. [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
2. [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier)
3. [Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

## Choosen machine learning algorithms
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classification

## Folder overview
The folder contains a mix of directories and files designed to organize the project's codebase, data, documentation, and operational workflows. This structure facilitates easy navigation and understanding of the project's purpose, how to set it up, and how it operates.

<u>Folder Structure</u>
1. .github: This directory houses GitHub Actions workflows. These workflows are automated processes that can be set to run on specific triggers, such as pushing code to the repository or manually triggering them. 
2. data: The data directory is intended to contain datasets used by the project. 
3. src: This folder contains 7 the Python scripts associated with the project. These scripts perform a variety of tasks, including the main script, data injestion, data preprocessing, cross validating, model training and performance evaluation
4. LICENSE.txt: This file contains the licensing information for the project. The license dictates how others can use, modify, and distribute the project. 
5. README.md: The README.md file is a Markdown file used to provide an overview of the project. It includes information on what the project does, how to set it up, how to use it, and any other relevant information that a user or contributor might need to know.
6. eda.ipynb: This Jupyter notebook contains exploratory data analysis (EDA). The notebook are used to explore and visualize the data, understand its structure, and derive insights that can guide further analysis and modeling.
7. requirements.txt: The requirements.txt file lists all the Python dependencies required to run the project. 
8. run.sh: This is a shell script file that contains commands to run the project. 

## EDA Overview
Summary of the EDA indicates four major input features for the models.
1. A large decrease in weight poses a higher risk of lung cancer
2. The presence of genetic markers poses a higher risk of lung cancer
3. Being female poses a higher risk of lung cancer
4. Higher exposure to air pollution poses a higher risk of lung cancer

Feature engineering
1. The 'smoke duration' was calculated by the difference between the 'Start Smoking' and 'Stop Smoking' years, with the *assumption that the dataset remains valid as patients who are still smoking are calculated using the current year.*
2. The 'weight change' was calculated by the difference between the 'Last weight' and 'Current weight' to observe weight changes. It was found in the Exploratory Data Analysis (EDA) that weight change has a higher correlation with the given features than others.
   
| Attribute                | Description                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| ID                       | Removed as found irrelevant.                                                                                    |
| Age                      | Outliers addressed. Negative ages were converted to absolute values. Removed as found irrelevant.                                                                           |
| Gender                   | Removed 1 'NaN' row, converted to nominal variables, and Z-scaled for the model.                                                                         |
| COPD History             | Used 'Taken Bronchodilators' to fill missing values (refer to EDA for justification). Removed as found irrelevant.                  |
| Genetic Markers          | Converted to a nominal variable and Z-scaled for the model.                    |
| Air Pollution Exposure   | Removed 3 'NaN' rows, converted to an ordinal variable, and Z-scaled for the model.                                  |
| Last Weight              | Data used to create a new feature called 'weight change'. Removed afterwards.                                                    |
| Current Weight           | Data used to create a new feature called 'weight change'. Removed afterwards.                                                 |
| Start Smoking            | Data used to create a new feature called 'smoke duration'. Removed afterwards.                                                          |
| Stop Smoking             | Data used to create a new feature called 'smoke duration'. Removed afterwards.                                                           |
| Taken Bronchodilators    | Utilized to fill in 'COPD History'. Removed afterwards due to strong correlation with 'COPD History'.                       |
| Frequency of Tiredness   | Converted to an ordinal variable. Removed as found irrelevant.                                               |
| Dominant Hand            | Removed as found irrelevant.                                                                  |
| Weight Change            | New feature from the weight difference between 'Last Weight' and 'Current Weight', and Z-scaled for the model. |
| Smoke Span               | New feature from the time difference between 'Start Smoking' and 'Stop Smoking'. Removed as found irrelevant. |
| Lung Cancer Occurrence   | Target. |

<u>Heatmap analysis before introducing to the algorithms</u>
Four weak features were identified and subsequently dropped to avoid introducing irrelevant features to the model<br>
![image](https://github.com/KeithChenYong/aiap-chen-yong-427J/assets/133010489/a65a0729-b4d8-4d94-b62c-3edd44fd98a7)

Refer to [eda.ipynb](https://github.com/KeithChenYong/aiap-chen-yong-427J/tree/main) for detailed analysis
