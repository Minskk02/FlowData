import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = pd.read_csv("target.csv")
classification = data['target']

continuousColumns = ['one', 'two', 'three', 'four']
continuousData = data[continuousColumns].astype(float)

#scale the continuous data
scaler = StandardScaler()
scaledData = scaler.fit_transform(continuousData)

tsne = TSNE(n_components=2, random_state=42, perplexity=2, learning_rate=200)
tsneResult = tsne.fit_transform(scaledData)


tsne_df = pd.DataFrame(tsneResult, columns=['t-sne component 1', 't-sne component 2'])
tsne_df['target'] = classification

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='t-sne component 1', y='t-sne component 2', hue='target',
                palette='Set1')

plt.title('perplexity = 2')
plt.show

# the red line which is laminar, is less curvy and forms tight datapoints, which makes 
#sense because of it's physical nature of flow - low variation and ease of modelling

#red and blue clusters are very well separated from each other, meaning this t-SNE model 
#separates each variables well with perplexity of 2

# blue lines look extremely curvy (hehe) and it means the feature variation is high
# and this can mean 2 things: it is highly dependent on Re number and(or) possibley highly non-linear!

features = ['one', 'two', 'three', 'four']
X = data[features]

k = 4
kmeans = KMeans(n_clusters = k, random_state = 42)
labels = kmeans.fit_predict(X)

data['Cluster'] = labels
print(data[['one', 'two', 'three', 'four', 'Cluster']])



