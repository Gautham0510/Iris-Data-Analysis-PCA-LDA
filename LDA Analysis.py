import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


import os
os.chdir("C:\Games")
iris_data = pd.read_csv("iris_data.csv")  
X = iris_data.iloc[:, :-1]  # Features - assuming the target is the last column
y = iris_data.iloc[:, -1]   # Target variable


le = LabelEncoder()
y_encoded = le.fit_transform(y)


lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y_encoded)

# Plotting the LDA components
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_encoded, cmap='viridis', alpha=0.7)
plt.title('Linear Discriminant Analysis')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.colorbar()
plt.show()
