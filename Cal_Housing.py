"""
feature engineering pipeline using skikit-learn 
(california housing price dataset from scikitlearn website)

"""

from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#intake data
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
#Registering the new columns from the exsisting column values. 
df = pd.DataFrame(X, columns=col_names)
df['MedInc_Log'] = np.log(df['MedInc'])
df['MedInc_Exp'] = np.exp(df['MedInc'])
df['HouseAge_Squared'] = df['HouseAge'] ** 2
df['Interaction'] = df['MedInc'] * df['AveRooms']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['Population_Normalized'] = scaler.fit_transform(df['Population'].values.reshape(-1, 1))

columns_to_keep = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'MedInc_Log', 'MedInc_Exp', 'HouseAge_Squared', 'Interaction']
correlation = df[columns_to_keep].corr() #.corr() is a function that measures the correlation from -1 to 1
print(correlation)

sns.heatmap(correlation, annot=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()



"""
[What is minmax scaling?]

MinMaxScaler() scales the values within the given range and 
.fit_transform computes 2 things at once: fit finds the min and max values of
 each column and transform applies scaling formula x' = ((x-x_min)/(x_max-x_min))
 
which in this code, the operaters assign the values inside the MinMaxScaler()

what this code does is just creating extra columns for future references and to 
see the correlations in between each major criterion and see if you can 
harvest meaningful insights out of it such as trends etc.,

"""
