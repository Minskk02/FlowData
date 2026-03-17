"""

"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import urllib.request
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/datasets/master/pima-indians-diabetes.data.csv"
with urllib.request.urlopen(url) as response:
    dataset = np.loadtxt(response,delimiter=',')
    
print(dataset[:5])

