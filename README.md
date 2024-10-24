Logistic Regression Classifier

This project demonstrates the use of Logistic Regression to predict survival on the Titanic dataset. It covers the full machine learning workflow, including data cleaning, preprocessing, model training, and evaluation.

Table of Contents

	•	Introduction
	•	Project Structure
	•	Installation
	•	Usage
	•	Dataset
	•	Model Training
	•	Evaluation
	•	Contributing
	•	License

Introduction

The goal of this project is to build a Logistic Regression model that predicts whether a passenger survived the Titanic disaster. This project walks through:

	•	Data preprocessing and cleaning
	•	Handling categorical features
	•	Splitting the data into training and testing sets
	•	Training a Logistic Regression model
	•	Evaluating model performance

Project Structure

logistic-classifier-project/
│
├── data/                # Dataset and related files
│   ├── raw/             # Raw dataset files (e.g., titanic.csv)
│   └── processed/       # Pre-processed data files (optional)
│
├── notebooks/           # Jupyter notebooks for experimentation and EDA
│   └── logistic_model.ipynb  
│
├── src/                 # Source code
│   ├── __init__.py      # Makes src a package
│   ├── data_loader.py   # Functions to load data
│   ├── train.py         # Model training script
│   └── evaluate.py      # Model evaluation script
│
├── models/              # Saved models and checkpoints
│   └── logistic_model.pkl  
│
├── reports/             # Reports and results
│   └── figures/         # Visualizations and plots
│
├── tests/               # Test cases for your code
│   └── test_train.py    # Unit tests for model training
│
├── LICENSE              # Apache 2.0 License
├── README.md            # Project README with details and instructions
├── requirements.txt     # Required Python packages for the project
├── .gitignore           # Files to ignore in version control
└── setup.py             # Setup script for easy installation

Installation

	1.	Clone the repository:

git clone https://github.com/AntexDSMLAI/logistic-regression-classifier.git
cd logistic-regression-classifier


	2.	Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


	3.	Install the required packages:

pip install -r requirements.txt


	4.	Launch Jupyter Notebook:

jupyter notebook notebooks/logistic_model.ipynb



Usage

	1.	Data Preprocessing:
The dataset is loaded from data/raw/titanic.csv, and the following preprocessing steps are applied:
	•	Fill missing values in the Age column with the median value
	•	Drop the Cabin column
	•	Encode categorical columns (Sex and Embarked) using LabelEncoder
	2.	Model Training and Evaluation:
	•	Split the dataset into training and test sets
	•	Train the model using Logistic Regression
	•	Evaluate the model with accuracy score, confusion matrix, and classification report

Dataset

The Titanic dataset contains the following features:

	•	PassengerId: Passenger identifier
	•	Survived: Survival status (0 = No, 1 = Yes)
	•	Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
	•	Name: Name of the passenger
	•	Sex: Gender of the passenger
	•	Age: Age of the passenger
	•	SibSp: Number of siblings/spouses aboard
	•	Parch: Number of parents/children aboard
	•	Ticket: Ticket number
	•	Fare: Passenger fare
	•	Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Model Training

The model is trained using Logistic Regression with the following script:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

Evaluation

The trained model is evaluated using:

	1.	Accuracy Score:

from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")


	2.	Confusion Matrix:

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_test_pred))


	3.	Classification Report:

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))



Contributing

Contributions are welcome! Please submit a pull request or open an issue to propose changes or improvements.

License

This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

Acknowledgments

	•	Kaggle Titanic Dataset
	•	scikit-learn Documentation
	•	Matplotlib and Seaborn for visualization

