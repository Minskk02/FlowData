# ML — Aerospace & Housing Data: Feature Engineering, Augmentation & Modelling

A collection of machine learning experiments spanning exploratory data analysis, feature engineering, data augmentation, dimensionality reduction, and model training. The datasets are primarily aerospace-themed (engine parameters, Mach/flow data) alongside the classic California Housing dataset.

---

## Files Overview

### 1. `ML_basic.py` — Exploratory Data Analysis (EDA)
**Dataset:** `Physically_Coupled_Engine_Dataset.csv`

Core EDA workflow on a physically coupled engine dataset:
- Load dataset and print shape, column names, head, and descriptive statistics
- Check for missing values per column
- Plot histograms for all columns
- Box plot + strip plot for the first column (exit velocity)
- **Outlier detection** using the IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR)
- **Pearson correlation matrix** visualised as a Seaborn heatmap
- Regression scatter plots for every feature vs. Exit Velocity

---

### 2. `Cal_Housing.py` — Feature Engineering Pipeline
**Dataset:** California Housing (scikit-learn built-in)

Demonstrates feature engineering techniques:
- Create derived features: `MedInc_Log`, `MedInc_Exp`, `HouseAge_Squared`, `Interaction` (MedInc × AveRooms)
- Normalise `Population` using **MinMaxScaler** (scales to [0, 1])
- Compute and visualise a **Pearson correlation heatmap** across original and engineered features to identify redundancy and useful interactions

---

### 3. `Cal_Housing_Lin_Reg.py` — Polynomial Linear Regression
**Dataset:** California Housing (scikit-learn built-in)

Applies polynomial regression and analyses feature importance:
- Engineer the same derived features as above
- Expand features with **PolynomialFeatures (degree = 3)**, generating higher-order and interaction terms
- Fit a **Linear Regression** model on the expanded feature set
- Rank and plot feature importances by absolute coefficient value (log-scale x-axis)
- Annotate the top 10 most important polynomial features

---

### 4. `Adding_Noise.py` — Data Augmentation, Encoding & Imbalance Handling
**Dataset:** `Ascending_Mach_Dataset_Imbalanced.csv`

A comprehensive data pre-processing and augmentation pipeline:

| Step | Description |
|------|-------------|
| **Velocity computation** | Derives velocity from Mach number and static temperature using `V = Mach × √(γRT)` |
| **Gaussian noise injection** | Adds Normal(0, σ=5) noise to velocity to simulate sensor uncertainty and increase data variety |
| **Bootstrapping (1000 resamples)** | Resamples the dataset with replacement 1000 times, plots the distribution of bootstrapped mean velocities to estimate sampling uncertainty |
| **Flow type classification** | Classifies each data point as Subsonic (<0.8), Transonic (0.8–1.1), or Supersonic (>1.1) Mach |
| **One-Hot Encoding** | Encodes the categorical `Flow_Type` column using `OneHotEncoder` |
| **Random Forest + Cross-Validation** | Trains a `RandomForestRegressor` with 3-fold cross-validation, reporting MSE and RMSE per fold |
| **Random Under-Sampling** | Uses `RandomUnderSampler` to balance the class distribution of flow types before training |
| **Train/Test Split** | Demonstrates `train_test_split` for dataset partitioning (67/33 split) |

> The file also contains inline notes explaining the distinction between data-preprocessing encoders and encoder networks in Variational Autoencoders (VAEs), and how back-propagation + gradient descent minimise the loss during weight initialisation.

---

### 5. `practice_paper.py` — Bootstrapping Practice
**Dataset:** `Aerospace_Specs_Dataset.csv`

A practice script focused on the bootstrapping concept:
- Sets up features `X` and target `y` (Velocity in m/s) from the aerospace specs dataset
- Documents the rationale for bootstrapping: validating dataset quality without collecting additional expensive data

---

### 6. `Training_Test.py` — Neural Network Binary Classifier
**Dataset:** Pima Indians Diabetes (fetched from GitHub via URL)

Builds a binary classifier using Keras:
- Fetches the Pima Indians Diabetes dataset directly via `urllib`
- Constructs a **Sequential Keras model** with Dense layers
- Evaluates using **confusion matrix**, **classification report**, and **ROC curve / AUC**

---

### 7. `tSNE.py` — Dimensionality Reduction & Clustering
**Dataset:** `target.csv` (flow classification data with laminar/turbulent labels)

Applies unsupervised learning techniques for visualisation and pattern discovery:
- **StandardScaler** normalises continuous features (`one`, `two`, `three`, `four`)
- **t-SNE** (perplexity = 2) reduces 4-D data to 2-D for visualisation
  - Laminar flow points form tight, low-variance clusters (physically expected)
  - Turbulent flow points are spread and curvy, indicating high feature variance and nonlinearity
- **KMeans clustering** (k = 4) groups the data into 4 clusters and appends cluster labels to the dataframe

---

## Datasets

| File | Description |
|------|-------------|
| `Physically_Coupled_Engine_Dataset.csv` | Engine parameters including exit velocity |
| `Aerospace_Specs_Dataset.csv` | Aerospace component specifications including velocity |
| `Ascending_Mach_Dataset_Imbalanced.csv` | Mach number and static temperature during ascent (class-imbalanced) |
| `Engine_Design_Parameters_Dataset.csv` | Engine design parameters |
| `Flat_Plate_Heat_Transfer_Data.csv` | Flat plate heat transfer data |
| `target.csv` | Flow classification data (laminar vs. turbulent) |

---

## Key Concepts Covered

- **EDA**: missing value checks, histograms, box plots, IQR outlier detection, correlation heatmaps
- **Feature Engineering**: log/exp/polynomial transforms, interaction terms, normalisation
- **Data Augmentation**: Gaussian noise injection, bootstrapping
- **Encoding**: One-Hot Encoding for categorical features
- **Class Imbalance**: Random under-sampling to equalise class distributions
- **Model Training**: Random Forest regression, Linear Regression, Keras Sequential NN
- **Evaluation**: Cross-validation (MSE/RMSE), confusion matrix, ROC/AUC
- **Unsupervised Learning**: t-SNE dimensionality reduction, KMeans clustering

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow / keras
```
