# Lung cancer classification
This project, led by Keith Chen Yong, aims to develop an end-to-end (E2E) solution for lung cancer classification.<br>

## Project Title
Prediction of Lung Cancer Using Classification Algorithms

##  Introduction
This initiative aims to predict the severity of lung cancer diagnosis in individuals based on personal information, sourced from Kaggle. The significance of early cancer detection cannot be overstated, as it is crucial for the timely administration of treatments that can significantly enhance survival rates. The core task of this project involves the application of classification algorithms to facilitate this prediction. Initially, three algorithms are employed: *Logistic Regression*, *Random Forest Classification*, and *Support Vector Machines* (SVM). These were selected for their proven effectiveness in similar tasks, with the structure of the code designed to accommodate the introduction of additional algorithms or hyperparameter fine-tuning of existing ones for optimal performance.

## Installation
All the necessary dependencies are specified in requirements.txt.

## Usage
Upon pulling or manually triggering the workflow in GitHub, the necessary installations from requirements.txt and the execution of the run.sh bash script will be completed. This will be followed by the execution of the main.py script. This Python script sequentially triggers various Python files responsible for tasks such as data ingestion (from a .db file named cancer_patient_data_sets within a folder titled data), data preprocessing, model building, k-fold cross-validation, and lastly, evaluation of the algorithms. Users will be able to see the k-fold cross-validation results, offering a comprehensive view of the dataset's behavior under different models and various classification performance metrics for each algorithm. <br><br>
![image](https://github.com/KeithChenYong/Lung-Cancer-Classification/assets/133010489/da1c3ed7-1397-4021-b2b3-572090d4b2f9)


User will be able to see the k-fold cross-validation results, offering a holistic view of the dataset's behavior under different models and various classification Performance metrics for each algorithm.<br>
**Note** Given the project's focus on the early stage detection of lung cancer, the <b>recall</b> metric (also known as sensitivity) should be prioritized to minimize the risk of overlooking actual stage of lung cancer. This prioritization underscores the project's commitment to ensuring the highest possible accuracy in identifying true positives.

## Configuration
Users have the flexibility to modify the following
1. Modify the k value in config.ini for cross-validation
2. Fine-tuning hyperparameters in config.ini to improve model performance or tailor it to different datasets
3. Introduce new algorithms in config.ini for exploring additional analytical approaches. *Note: Additional of new algorithms will require modify of model.py. Please contact author for further guidance, contact information listed below*

| ML Algorithm            | Justification |
|-------------------------|---------------|
| Logistic Regression     | - Simple, fast, and efficient for linearly separable data.<br>- Good baseline model. |
| Random Forest Classifier | - Handles non-linear data well due to ensemble of decision trees.<br>- Robust to overfitting with large number of features.<br>- Less sensitive to class imbalance (provided data are not imbalance). |
| Support Vector Classification | - Effective in high-dimensional spaces.<br>- Kernel trick allows for non-linear classification.<br>- Soft margin approach enhances generalization. |

## Contributing
Contributions are welcome! Please submit bug reports or feature requests via the [GitHub issue tracker](https://github.com/KeithChenYong/Lung-Cancer-Classification/issues).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details. 

## Credits
**Kaggle - The Devastator (Author)** as the [source](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data) of data provision. 

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
Summary of the EDA indicates that mutiple features exhibits high corelation between features as shown by the heatmap below using 'spearman' method. <br>
![image](https://github.com/KeithChenYong/Lung-Cancer-Classification/assets/133010489/cc8abe25-f969-43ab-bc34-9679ef4ccee6)


At first glance, this dataset might appear suitable. However, its cleaned and pre-processed nature restricts our capability to conduct feature engineering. A violin plot shows us that all features does indeed provide some insights to the prediction of cancer severity. <br>
![image](https://github.com/KeithChenYong/Lung-Cancer-Classification/assets/133010489/6b8daa9e-4f09-4529-a0b3-abf0a3936023)


In order to use all feature that are correlated, Principal Component Analysis (PCA) was utilized to reduce dimensionality and decorrelate features for the models. The Kaiser Criterion was applied, selecting 6 components. PCA serves to reduce overfitting and computational complexity by eliminating correlated features and reducing dimensionality. While this aids in reducing training time and computational costs, it introduces bias and leads to less interpretable models, often resulting in some loss of information.<br>
![image](https://github.com/KeithChenYong/Lung-Cancer-Classification/assets/133010489/ea01c9db-27a6-4c7b-941e-ec493e8f9722)
<br>
![image](https://github.com/KeithChenYong/Lung-Cancer-Classification/assets/133010489/e3544687-ea4c-4fae-bdc3-647728a29f97)


Refer to [eda.ipynb](https://github.com/KeithChenYong/Lung-Cancer-Classification) for detailed analysis
