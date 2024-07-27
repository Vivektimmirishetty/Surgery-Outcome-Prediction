# Surgery-Outcome-Prediction
Project Overview
Welcome to the Surgery Outcome Prediction project! This project aims to predict the outcome of surgeries (either "Abort" or "Success") based on various patient factors and medical data. Using machine learning algorithms, we analyze and classify the surgery outcomes to assist medical professionals in decision-making.

Features
Data Visualization: Visualize the distribution of surgery outcomes and other features.
Data Preprocessing: Handle missing values, encode categorical variables, and scale features.
Model Training: Train and evaluate machine learning models, including Naive Bayes and Random Forest classifiers.
Model Evaluation: Assess model performance using accuracy, precision, recall, F1-score, and confusion matrix.
Outcome Prediction: Predict surgery outcomes and display the results.
Dataset
The dataset used in this project is postoperativedata.csv, which contains the following columns:

L-CORE
L-SURF
L-O2
L-BP
SURF-STBL
CORE-STBL
BP-STBL
decision ADM-DECS
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/surgery-outcome-prediction.git
cd surgery-outcome-prediction
Install Required Libraries:
Ensure you have Python installed. Then, install the required libraries:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Load the Data:

Load the dataset from postoperativedata.csv.
python
Copy code
dataset = pd.read_csv("postoperativedata.csv")
Data Visualization:

Visualize the distribution of surgery outcomes.
python
Copy code
plt.figure(figsize=(8,6))
ax = sns.countplot(data=dataset, x='decision ADM-DECS')
plt.show()
Data Preprocessing:

Handle missing values, encode categorical variables, and scale features.
python
Copy code
# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)
X['L-CORE'] = le.fit_transform(X['L-CORE'])
# Continue for other columns
Model Training and Evaluation:

Train Naive Bayes and Random Forest classifiers and evaluate their performance.
python
Copy code
# Naive Bayes Classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
python
Copy code
# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=200, random_state=24, criterion='entropy', max_depth=10, min_samples_split=3)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
Model Evaluation:

Assess model performance using accuracy, precision, recall, F1-score, and confusion matrix.
python
Copy code
print("Accuracy:", accuracy_rf)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
Outcome Prediction:

Predict surgery outcomes and display the results.
python
Copy code
predict = random_forest.predict(X_test)
for i in range(len(predict)):
    if predict[i] == 0:
        print("{} :*********************************** Abort ".format(X_test.iloc[i,:]))
    else:
        print("{} :*********************************** Success ".format(X_test.iloc[i,:]))
Contribution
We welcome contributions to improve the Surgery Outcome Prediction project. Feel free to fork this repository, submit issues, and send pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
Author: Timmirishetty Vivek
Email: timmirishettyvivek@gmail.com
LinkedIn: Vivek Timmirishetty
