import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
os.chdir("C:\Games")
file_path = "iris_data.csv"
iris_data = pd.read_csv(file_path)

# Assuming your target column is the last column
target = iris_data.iloc[:, -1]
features = iris_data.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
scaledup_feature = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(scaledup_feature)

# Visualize the explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='Cumulative explained variance')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# The transformed data using principal components
main_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

result = pd.concat([main_df, target], axis=1)

# Display the final data with principal components
print(result)
