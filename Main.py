
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracyscore, classification_report,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
# Load the generated data
dataset=pd.read_csv("postoperativedata.csv")
dataset.head()
dataset.shape

# Create a count plot for the label column
plt.figure(figsize=(8,6))
ax = sns.countplot(data=dataset, x='decision ADM-DECS')
plt.xlabel('decision ADM-DECS')
plt.ylabel('Count')
plt.title('Count Plot')

# Annotate bars with counts
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() 
/ 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
#checking null values
dataset.isnull().sum()
# Split the data into features and labels
X = dataset.drop(['decision ADM-DECS'], axis=1)
X
y = dataset['decision ADM-DECS']
y
#Label encoding
le= LabelEncoder()
y=le.fit_transform(y)
X['L-CORE'] = le.fit_transform(X['L-CORE'])
X['L-SURF'] = le.fit_transform(X['L-SURF'])
X['L-O2'] = le.fit_transform(X['L-O2'])
X['L-BP'] = le.fit_transform(X['L-BP'])
X['SURF-STBL'] = le.fit_transform(X['SURF-STBL'])
X['CORE-STBL'] = le.fit_transform(X['CORE-STBL'])
X['BP-STBL'] = le.fit_transform(X['BP-STBL'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#naive bayes
# Naive Bayes Classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)

# Evaluation
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='macro')
recall_nb = recall_score(y_test, y_pred_nb, average='macro')
f1_nb = f1_score(y_test, y_pred_nb, average='macro')

print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_nb)
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1-Score:", f1_nb)
#Random_forest
random_forest=RandomForestClassifier(n_estimators=200, random_state=24,criterion='entropy',max_depth=10,min_samples_split=3)
random_forest.fit(X_train,y_train)
y_pred=random_forest.predict(X_test)
y_pred
ac=accuracy_score(y_test,y_pred)*100
ac
cm=confusion_matrix(y_test,y_pred)
cm
report=classification_report(y_test,y_pred)
print(report)
fig, ax = plt.subplots()

# Create a heatmap using seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set labels and title
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Random_forest_Confusion Matrix')

# Show the plot
plt.show()

#prediction
A='Abort'
S='Success'

predict = random_forest.predict(X_test)
for i in range(len(predict)):
    if predict[i] == 0:
        print("{} :***********************************{} ".format(X_test.iloc[i,:],A))
    else:
        print("{} :***********************************{} ".format(X_test.iloc[i,:],S))
