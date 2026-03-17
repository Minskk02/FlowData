
"""
applying linear regression on the Cal_Housing data

"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#intake data
data = fetch_california_housing(as_frame=True)
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
#Registering the new columns from the exsisting column values. 
df = pd.DataFrame(X, columns=col_names)
df['MedInc_Log'] = np.log(df['MedInc'])
df['MedInc_Exp'] = np.exp(df['MedInc'])
df['HouseAge_Squared'] = df['HouseAge'] ** 2
df['Interaction'] = df['MedInc'] * df['AveRooms']

poly = PolynomialFeatures(degree = 3, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = [f'feature_{i}' for i in 
range(X_poly.shape[1])]

X_poly_df = pd.DataFrame(X_poly, columns = feature_names)

model = LinearRegression()
model.fit(X_poly_df, y)

importances = np.abs(model.coef_)

sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]

plt.figure(figsize=(10,6))
plt.bar(range(len(sorted_importances)), sorted_importances)
plt.xticks(np.arange(0, len(sorted_importances), 5), np.arange(1, len(sorted_importances)+1, 5), rotation = 'vertical')

plt.xlabel('Feature Number')
plt.ylabel('Importances')
plt.title('Feature Importances')


top_features = range(1,11)
top_importances = sorted_importances[:10]

max_importance = np.max(sorted_importances)

label_x = len(sorted_importances)+1
label_y = max_importance + 0.03

for i, (feature, importance) in enumerate(zip(top_features, top_importances)):
    plt.text(
        label_x, label_y - i * (max_importance / 10), 
        f'Feature {feature}: {importance:.4f}',
        ha = 'right'    
    )

plt.xscale('log')
plt.tight_layout()
plt.show()



"""
variances = np.var(X_poly, axis = 0)
sorted_indices = np.argsort(variances)[::-1]
sorted_variances = variances[sorted_indices]

plt.figure(figsize=(10,6))
plt.bar(range(len(sorted_variances)), sorted_variances)
plt.xticks(np.arange(len(sorted_variances)), feature_names, rotation='vertical')

plt.xscale('log')
plt.yscale('log')

top_feature_variance = sorted_variances[0]
threshold_99 = np.percentile(sorted_variances, 99)

count_less_than_99 = np.sum(sorted_variances < threshold_99)

plt.show()

"""








