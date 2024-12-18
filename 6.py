# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('iris.csv')

# Display dataset information
print(dataset.describe())
print(dataset.info())

# Splitting the dataset into Training set and Test set
X = dataset.iloc[:, :-1].values  # Features (excluding the last column)
y = dataset.iloc[:, -1].values   # Target (last column)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Predict probabilities
probs = classifier.predict_proba(X_test)
probs_y = np.round(probs, 2)

# Results display
res = "{:<10} | {:<10} | {:<5}".format("y_test", "y_pred", "Setosa(%)", "versicolor(%)", "virginica(%)\n")
for i in range(len(y_test)):
    res += "{:<10} | {:<10} | {:<5} | {:<5} | {:<5}\n".format(y_test[i], y_pred[i], probs_y[i][0]*100, probs_y[i][1]*100, probs_y[i][2]*100)

print(res)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
ax = plt.axes()
df_cm = pd.DataFrame(cm, index=['Setosa', 'Versicolor', 'Virginica'], columns=['Setosa', 'Versicolor', 'Virginica'])
sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 20}, ax=ax)
ax.set_title('Confusion Matrix')
plt.show()
