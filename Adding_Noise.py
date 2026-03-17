"""
adding noise in the dataset to create sort of reference points for 
the machine to learn better, increasing the datapoints to macroscopically 
increase the accuracy

using LLMs for dataset generation is a charming option to see the 
model accuracy or generalisaiton of the model. this is the test
of that use case of LLMs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
df = pd.read_csv("Ascending_Mach_Dataset_Imbalanced.csv")
print(df.columns.tolist())

gamma = 1.401
R = 287.15
a_check = (gamma * R * df['Static_T_K'])**0.5
V_check = df['Mach'] * a_check
df['Velocity'] = V_check

def add_noise_to_velocity(df, noise_level = 5):
    df = df.copy()
    noise = np.random.normal(0, noise_level, df['Velocity'].shape)
    df['Velocity'] += noise
    return df

noisy_df = add_noise_to_velocity(df)

df['Sample'] = np.arange(len(df))
noisy_df['Sample'] = np.arange(len(noisy_df))

plt.figure(figsize=(10, 5))
plt.plot(df['Sample'], df['Velocity'], label='Original Velocity', color='blue', linewidth=2)
plt.plot(noisy_df['Sample'], noisy_df['Velocity'], label='Noisy Velocity', color='red', alpha=0.6)

plt.title('Velocity vs Sample Number (Before and After Noise)')
plt.xlabel('Sample Number')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.show()

X=df.copy()
y = X['Velocity']
X_features = X.drop('Velocity', axis = 1)

def bootstrap_sample(X, y):
    indices = np.random.choice(X.shape[0], size = X.shape[0], replace=True)
    return X.iloc[indices], y.iloc[indices]

X_bootstrapped, y_bootstrapped = bootstrap_sample(X_features, y)

plt.figure(figsize=(10,5))
sns.histplot(y, bins=30, color='blue', label='Original Velocity', alpha=0.6, kde=True)
sns.histplot(y_bootstrapped, bins=30, color='red', label='Bootstrapped Velocity', alpha=0.4, kde=True)

plt.xlabel('Velocity (m/s)')
plt.ylabel('Frequency')
plt.title('Distribution of Velocity: Original vs Bootstrapped')
plt.legend()
plt.grid(True)
plt.show()
"""
bootstrapping one time is not the main purpose of it.
instead, perform at least 1000 times (resampling) and 
plot the histogram of the means or any other variables in concern
and it is shown below. 
"""
boot_means = []

for i in range(1000):  # 1000 bootstrap resamples
    _, yb = bootstrap_sample(X_features, y)
    boot_means.append(yb.mean())  # compute statistic per resample

# Now plot the distribution of means
sns.histplot(boot_means, bins=30, kde=True, color='purple')
plt.xlabel('Bootstrapped Mean Velocity (m/s)')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of the Mean Velocity')
plt.show()
    
"""
this case, shows the mean of the values you'll likely get
from how you would normally resample from a population.
This is the true power of bootstrapping.
"""
#Encoding...
if 'Flow_Type' not in df.columns:
    def classify_flow_type(mach):
        if mach < 0.8:
            return 'Subsonic'
        elif mach < 1.1:
            return 'Transonic'
        else:
            return 'Supersonic'
    df['Flow_Type'] = df['Mach'].apply(classify_flow_type)

X_features = df.drop('Velocity', axis=1)
y = df['Velocity']
    
  # one-hot encode the "flow_type" column
encoder = OneHotEncoder(sparse_output=False,
                        handle_unknown = 'ignore')  
encoded_features = encoder.fit_transform(X_features[['Flow_Type']])
X_feature_encoded = pd.concat([X_features.drop(columns=['Flow_Type']), 
                               pd.DataFrame(encoded_features, 
                                            columns=encoder.get_feature_names_out(['Flow_Type']), 
                                            index=X_features.index)], axis=1)

model_2 = RandomForestRegressor(random_state=42)
scores = cross_val_score(model_2,X_feature_encoded, y, 
                         cv=3, scoring='neg_mean_squared_error')

mse_per_fold  = -scores
rmse_per_fold = np.sqrt(mse_per_fold)

print("MSE per fold :", mse_per_fold)
print("RMSE per fold:", rmse_per_fold)
print("Mean RMSE    :", rmse_per_fold.mean())

"""
well what does these stats mean? --> encoding comes after feature engineering!
feature engineering adds new useful colmns containing new variables 
and encoders convert those features into a machine_readable numbers 
before the training and becomes the part of the training vector.

BTW encoder in VAE means something different.
--> it is essentially data modifying method, but in VAE NN, 
it is more for reducing super high dimension dataset such as images
into lower ones by compresssing it using latent vector
 (e.g. down to 32-D level)
and it happens at a very different level of the work flow than typical
data pre-processing stage.
input -> encoder network -> latent vector -> decoder network -> reconstructed output

"""   
    
rus = RandomUnderSampler(random_state = 42)

X_downsampled, y_downsampled = rus.fit_resample(X_features, df['Flow_Type'])
# what the fuck is the purpose of this? 
# why make the numbers of classes the same?
"""
^^^^ answering that, essensially we need to undersample a class that is much larger in 
sample size to match the number of the other classes for fairness when it comes to
the training output since ML is likely to guess whichever group that is abundant in count
This could be the last step before the training stage.
"""
# BEFORE
print("Before (counts):")
print(df['Flow_Type'].value_counts())
print("\nBefore (percent):")
print((df['Flow_Type'].value_counts(normalize=True)*100).round(2))

# AFTER
print("\nAfter (counts):")
print(y_downsampled.value_counts())
print("\nAfter (percent):")
print((y_downsampled.value_counts(normalize=True)*100).round(2))

"""
now it's almost ready for the training process. now it's the loop of 
TRAINING->EVALUATION->IMPROVEMENT... except one thing maybe = dataset partitioning.

Then pick the model that performed the best and do blind evaluation. 

"""

from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5,2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""
the code block above is performing "Dataset Partitioning"
--> 


So y_train acts almost like an encoder but the fact that it contains the "answer sheet" 
for the X's feature training process to adjust the weight of each row or layers of neurons 
during the tuning stage? --> almost correect

y_train acts as answer sheet but that is done in the Loss function calc
when initialising the weights. Then It undergoes "Adjustment", using
"back-propagation" and "gradient descend" to minimise the loss during
the weight initialisation stage (MSE). This is essentially how training 
and testing ML model works.
"""











