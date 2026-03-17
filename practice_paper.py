import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
"""
Practice paper
"""

"""
why do we perform bootstrapping?
->Bootstrapping is a useful tool where you can process and 
test if the dataset you have is accurate enough without
needing to get more data. This is especially crucial when
extracting extra data is expensive or time consuming.
Here is how it's done.
"""
# import data first
data = pd.read_csv("Aerospace_Specs_Dataset.csv")
X = data.copy()
y = X['Velocity (m/s)']
X_features = X.drop('Velocity (m/s)', axis=1) 
# here, axis=1 means column direction, where axis=0 menas rows
