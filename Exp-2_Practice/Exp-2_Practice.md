## In-Lab : Data Preprocessing and Feature Engineering for Machine Learning

### Task 1: Implement data preprocessing techniques including handling missing values, normalization, standardization, and data transformation (using Pima Indians diabetes Dataset).


```python
import numpy as np
import pandas as pd

# Load Pima Indians Diabetes dataset from public repository
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
pima_df = pd.read_csv(url, header=None, names=columns)


print("Dataset Shape (rows, cols):", pima_df.shape)
print("\nFirst 3 rows of raw dataset:")
print(pima_df.head(3))

# Count invalid zero values in features where a zero is physiologically impossible
zero_counts = (pima_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
print("\nInvalid Zero Values per column:\n", zero_counts)

```

    Dataset Shape (rows, cols): (768, 9)
    
    First 3 rows of raw dataset:
       Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    1            1       85             66             29        0  26.6   
    2            8      183             64              0        0  23.3   
    
       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    1                     0.351   31        0  
    2                     0.672   32        1  
    
    Invalid Zero Values per column:
     Glucose            5
    BloodPressure     35
    SkinThickness    227
    Insulin          374
    BMI               11
    dtype: int64
    


```python
from sklearn.impute import SimpleImputer

# Replace invalid zero values with NaN
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
pima_df[cols_to_impute] = pima_df[cols_to_impute].replace(0, np.nan)

print("Missing values after marking zeros:\n", pima_df.isnull().sum())

# Apply median imputation to replace NaN values
imputer = SimpleImputer(strategy='median')
pima_df[cols_to_impute] = imputer.fit_transform(pima_df[cols_to_impute])

print("\nMissing values after statistical median imputation:\n", pima_df.isnull().sum())

```

    Missing values after marking zeros:
     Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64
    
    Missing values after statistical median imputation:
     Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64
    


```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

features = columns[:-1]
X = pima_df[features]

# Apply MinMaxScaler (Normalization)
min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)
print("Normalized Range Boundaries (Min/Max):", X_normalized.min(), "to", X_normalized.max())

# Apply StandardScaler (Standardization)
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
print("Standardized Mean (approx 0):", round(X_standardized.mean(), 4))
print("Standardized Std Dev (approx 1):", round(X_standardized.std(), 4))

```

    Normalized Range Boundaries (Min/Max): 0.0 to 1.0000000000000002
    Standardized Mean (approx 0): 0.0
    Standardized Std Dev (approx 1): 1.0
    


```python
from sklearn.preprocessing import PowerTransformer

# Evaluate raw skewness of the Insulin column
original_skew = pima_df['Insulin'].skew()
print("Original skewness coefficient of 'Insulin':", round(original_skew, 4))

# Apply Yeo-Johnson transformation to stabilize variance
power_transformer = PowerTransformer(method='yeo-johnson')
insulin_transformed = power_transformer.fit_transform(pima_df[['Insulin']])

# Check the new skewness coefficient
transformed_skew = pd.Series(insulin_transformed.flatten()).skew()
print("Skewness coefficient after Power Transformation:", round(transformed_skew, 4))

```

    Original skewness coefficient of 'Insulin': 3.38
    Skewness coefficient after Power Transformation: 0.0276
    


```python
# Create preprocessed DataFrame combining standardized features and outcomes
processed_pima_df = pd.DataFrame(data=X_standardized, columns=features)
processed_pima_df['Outcome'] = pima_df['Outcome']

# Export to system directory
processed_pima_df.to_csv("processed_pima_diabetes.csv", index=False)
print("Processed Pima dataset exported successfully to 'processed_pima_diabetes.csv'. Shape:", processed_pima_df.shape)

```

    Processed Pima dataset exported successfully to 'processed_pima_diabetes.csv'. Shape: (768, 9)
    

### Task 2: Apply feature scaling and dimensional transformation techniques to improve model readiness (Sonar Dataset)


```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load Sonar dataset
sonar_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
sonar_features = [f"F_{i}" for i in range(1, 61)]
sonar_columns = sonar_features + ['Class']
sonar_df = pd.read_csv(sonar_url, header=None, names=sonar_columns)

# Encode class: M (Mine) -> 1, R (Rock) -> 0
sonar_df['Class_numeric'] = sonar_df['Class'].map({'M': 1, 'R': 0})

X_sonar = sonar_df[sonar_features]

# Scale features
scaler = StandardScaler()
X_sonar_scaled = scaler.fit_transform(X_sonar)

print("Standardized features shape:", X_sonar_scaled.shape)


```

    Standardized features shape: (208, 60)
    


```python
from sklearn.decomposition import PCA

# Apply PCA to preserve 90% of total variance
pca_90 = PCA(n_components=0.90, random_state=42)
X_pca_90 = pca_90.fit_transform(X_sonar_scaled)

print("Original dimensions count:", X_sonar_scaled.shape[1])
print("Reduced dimensions count (to preserve 90% variance):", X_pca_90.shape[1])

```

    Original dimensions count: 60
    Reduced dimensions count (to preserve 90% variance): 22
    


```python
import matplotlib.pyplot as plt
import numpy as np

# Fit full PCA to compute complete variance ratios
pca_full = PCA(random_state=42)
pca_full.fit(X_sonar_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot Cumulative Explained Variance
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
plt.title("Scree Plot: Cumulative Explained Variance Ratio 24EU01121")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.legend()
plt.show()

```


    
![png](output_10_0.png)
    



```python
from sklearn.manifold import TSNE
import seaborn as sns

# Project standardized features onto a 2D space using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_sonar_scaled)

# Construct visualization DataFrame
tsne_df = pd.DataFrame(X_tsne, columns=['Dim_1', 'Dim_2'])
tsne_df['Class'] = sonar_df['Class']

# Plot t-SNE clusters
plt.figure(figsize=(6, 4))
sns.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2', hue='Class', palette='Set1')
plt.title("t-SNE Projection Map of Sonar Feature Space 24EU01121")
plt.show()

```


    
![png](output_11_0.png)
    



```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Evaluate performance on all 60 scaled raw features
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_sonar_scaled, sonar_df['Class_numeric'], test_size=0.3, random_state=42)
lr_raw = LogisticRegression()
lr_raw.fit(X_train_raw, y_train)
acc_raw = accuracy_score(y_test, lr_raw.predict(X_test_raw))

# Evaluate performance on the reduced PCA components
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca_90, sonar_df['Class_numeric'], test_size=0.3, random_state=42)
lr_pca = LogisticRegression()
lr_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, lr_pca.predict(X_test_pca))

print(f"Accuracy using all 60 scaled features: {acc_raw * 100:.2f}%")
print(f"Accuracy using {X_pca_90.shape[1]} PCA components:  {acc_pca * 100:.2f}%")

```

    Accuracy using all 60 scaled features: 76.19%
    Accuracy using 22 PCA components:  87.30%
    

### Task 3: Evaluate feature importance and implement feature selection techniques for enhanced predictive performance (using Pima Indians Diabetes Dataset).


```python
from sklearn.feature_selection import SelectKBest, f_classif

# Rank and select the top 4 features using SelectKBest (ANOVA F-value)
selector = SelectKBest(score_func=f_classif, k=4)
selector.fit(X_standardized, pima_df['Outcome'])

# Build a summary DataFrame of feature scores
anova_scores = pd.DataFrame({'Feature': features, 'F-Score': selector.scores_}).sort_values(by='F-Score', ascending=False)
print("ANOVA F-Test Scores:\n", anova_scores)

```

    ANOVA F-Test Scores:
                         Feature     F-Score
    1                   Glucose  245.667855
    5                       BMI   82.629271
    7                       Age   46.140611
    0               Pregnancies   39.670227
    3             SkinThickness   37.078538
    4                   Insulin   33.190796
    6  DiabetesPedigreeFunction   23.871300
    2             BloodPressure   21.631580
    


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Run RFE using Logistic Regression to select the top 4 features
estimator = LogisticRegression()
rfe = RFE(estimator=estimator, n_features_to_select=4)
rfe.fit(X_standardized, pima_df['Outcome'])

# Build selection ranking summary
rfe_support = pd.DataFrame({'Feature': features, 'Selected': rfe.support_, 'Ranking': rfe.ranking_}).sort_values(by='Ranking')
print("RFE Selected Features:\n", rfe_support)

```

    RFE Selected Features:
                         Feature  Selected  Ranking
    0               Pregnancies      True        1
    1                   Glucose      True        1
    5                       BMI      True        1
    6  DiabetesPedigreeFunction      True        1
    7                       Age     False        2
    2             BloodPressure     False        3
    4                   Insulin     False        4
    3             SkinThickness     False        5
    


```python
from sklearn.linear_model import LogisticRegression

# Run Logistic Regression with Lasso (L1) regularization
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
lasso.fit(X_standardized, pima_df['Outcome'])

# Display feature coefficients (zero-coefficients indicate dropped features)
lasso_coefs = pd.DataFrame({'Feature': features, 'Coefficient': lasso.coef_[0]}).sort_values(by='Coefficient', key=abs, ascending=False)
print("Lasso Regularization Coefficients:\n", lasso_coefs)

```

    Lasso Regularization Coefficients:
                         Feature  Coefficient
    1                   Glucose     1.106412
    5                       BMI     0.611338
    0               Pregnancies     0.399515
    6  DiabetesPedigreeFunction     0.269034
    7                       Age     0.134767
    2             BloodPressure    -0.073190
    4                   Insulin    -0.067230
    3             SkinThickness     0.019400
    


```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Train Random Forest to compute Gini feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X, pima_df['Outcome']) # Random Forest is scale invariant; raw values are suitable

# Plot feature importances
importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importances:\n", importances)

plt.figure(figsize=(6, 4))
sns.barplot(data=importances, x='Importance', y='Feature', palette='crest')
plt.title("Random Forest Feature Importances 24EU01121")
plt.show()

```

    Random Forest Feature Importances:
                         Feature  Importance
    1                   Glucose    0.263715
    5                       BMI    0.167470
    7                       Age    0.127208
    6  DiabetesPedigreeFunction    0.124075
    4                   Insulin    0.084843
    2             BloodPressure    0.082722
    0               Pregnancies    0.079268
    3             SkinThickness    0.070699
    


    
![png](output_17_1.png)
    



```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split dataset into training and test sets
X_train, X_test, y_train_p, y_test_p = train_test_split(X, pima_df['Outcome'], test_size=0.3, random_state=42)

# Model 1: Train on all 8 features
rf_full = RandomForestClassifier(random_state=42)
rf_full.fit(X_train, y_train_p)
preds_full = rf_full.predict(X_test)

# Model 2: Train on only the top 4 features identified by Random Forest
top_4_cols = importances.head(4)['Feature'].tolist()
X_train_sub = X_train[top_4_cols]
X_test_sub = X_test[top_4_cols]

rf_sub = RandomForestClassifier(random_state=42)
rf_sub.fit(X_train_sub, y_train_p)
preds_sub = rf_sub.predict(X_test_sub)

print("--- CLASSIFICATION REPORT: ALL FEATURES (8 features) ---")
print(classification_report(y_test_p, preds_full))

print("\n--- CLASSIFICATION REPORT: FEATURE-SELECTED SUBSET (Top 4 features) ---")
print(classification_report(y_test_p, preds_sub))

```

    --- CLASSIFICATION REPORT: ALL FEATURES (8 features) ---
                  precision    recall  f1-score   support
    
               0       0.82      0.79      0.81       151
               1       0.63      0.66      0.65        80
    
        accuracy                           0.75       231
       macro avg       0.72      0.73      0.73       231
    weighted avg       0.75      0.75      0.75       231
    
    
    --- CLASSIFICATION REPORT: FEATURE-SELECTED SUBSET (Top 4 features) ---
                  precision    recall  f1-score   support
    
               0       0.82      0.76      0.79       151
               1       0.60      0.68      0.64        80
    
        accuracy                           0.73       231
       macro avg       0.71      0.72      0.71       231
    weighted avg       0.74      0.73      0.73       231
    
    

## Post-Lab : Perform preprocessing on Titanic Dataset


```python
import pandas as pd
import numpy as np

# Load Titanic dataset from GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

titanic_df = pd.read_csv(url)

print("First 5 Rows:")
display(titanic_df.head())

print("\nDataset Shape:", titanic_df.shape)

print("\nDataset Information:")
titanic_df.info()
```

    First 5 Rows:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


    
    Dataset Shape: (891, 12)
    
    Dataset Information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    


```python
# Check missing values
print("Missing Values:")
print(titanic_df.isnull().sum())
```

    Missing Values:
    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    


```python
# Fill missing Age with median
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())

# Fill missing Embarked with mode
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

# Drop Cabin because most values are missing
titanic_df.drop(columns=['Cabin'], inplace=True)

print("Missing Values After Preprocessing:")
print(titanic_df.isnull().sum())
```

    Missing Values After Preprocessing:
    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    dtype: int64
    


```python
# Convert Sex into numerical values
titanic_df['Sex'] = titanic_df['Sex'].map({'male':0, 'female':1})

# One-Hot Encoding for Embarked
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], drop_first=True)

print("Data Types:")
print(titanic_df.dtypes)
```

    Data Types:
    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex              int64
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Embarked_Q        bool
    Embarked_S        bool
    dtype: object
    


```python
# Remove unnecessary columns
titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print("Dataset Shape:", titanic_df.shape)

display(titanic_df.head())
```

    Dataset Shape: (891, 9)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



```python
duplicates = titanic_df.duplicated().sum()

print("Duplicate Rows:", duplicates)

titanic_df.drop_duplicates(inplace=True)

print("Final Shape:", titanic_df.shape)
```

    Duplicate Rows: 116
    Final Shape: (775, 9)
    


```python
print("Preprocessed Titanic Dataset")

display(titanic_df.head())

print("\nMissing Values:")
print(titanic_df.isnull().sum())

print("\nData Types:")
print(titanic_df.dtypes)
```

    Preprocessed Titanic Dataset
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing Values:
    Survived      0
    Pclass        0
    Sex           0
    Age           0
    SibSp         0
    Parch         0
    Fare          0
    Embarked_Q    0
    Embarked_S    0
    dtype: int64
    
    Data Types:
    Survived        int64
    Pclass          int64
    Sex             int64
    Age           float64
    SibSp           int64
    Parch           int64
    Fare          float64
    Embarked_Q       bool
    Embarked_S       bool
    dtype: object
    


```python

```
