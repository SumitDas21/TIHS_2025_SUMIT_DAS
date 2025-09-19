Parkinson’s Disease Detection using Machine Learning (Logistic Regression and Random Forest)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


" QUESTION 1 :
From the scatterplot/pairplot, which two features seem most useful for separating the classes?
"


%pip install ucimlrepo

from ucimlrepo import fetch_ucirepo, list_available_datasets

# check which datasets can be imported
list_available_datasets()

# import dataset
heart_disease = fetch_ucirepo(id=45)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = heart_disease.data.features
y = heart_disease.data.targets
# train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

# access metadata
print(heart_disease.metadata.uci_id)
print(heart_disease.metadata.num_instances)
print(heart_disease.metadata.additional_info.summary)

# access variable info in tabular format
print(heart_disease.variables)



%pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
heart = fetch_ucirepo(id=45)

# Features and target
X = heart.data.features
y = heart.data.targets
if isinstance(y, pd.DataFrame):
    y = y.rename(columns={y.columns[0]: "target"})
else:
    y = pd.DataFrame(y, columns=["target"])

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)

# Quick look
print(df.head())
print(df.describe())

# Pairplot with hue as target
sns.pairplot(df[['age','chol','trestbps','thalach','target']], hue="target")
plt.show()

from ucimlrepo import fetch_ucirepo

# Fetch Parkinson's dataset
parkinsons = fetch_ucirepo(id=174)

# Access data
X_parkinsons = parkinsons.data.features
y_parkinsons = parkinsons.data.targets

# Combine into a DataFrame
df_parkinsons = pd.concat([X_parkinsons, y_parkinsons], axis=1)

# Display the first few rows and info
print(df_parkinsons.head())
df_parkinsons.info()


"
QUESTION 2 :
Looking at the correlation heatmap, which pair of features are most correlated? What might this imply?
"

from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Parkinson's dataset from UCI (ID = 174)
parkinsons = fetch_ucirepo(id=174)

# Features and target
X = parkinsons.data.features
y = parkinsons.data.targets

# Combine for visualization
df = pd.concat([X, y], axis=1)

# Display first rows
print(df.head())


"
QUESTION 3 :
Why do we split the dataset into training and testing sets?
"


from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load Parkinson's dataset
parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features
y = parkinsons.data.targets
df = pd.concat([X, y], axis=1)

# Step 2: Compute correlation matrix (only numeric features)
numeric_df = df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()

# Step 3: Plot correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Parkinson's Dataset")
plt.show()

# Step 4: Find the most strongly correlated pair (excluding self-correlation)
corr_matrix_abs = corr_matrix.abs()
np.fill_diagonal(corr_matrix_abs.values, 0)  # ignore self-correlation
max_corr = corr_matrix_abs.stack().idxmax()
max_value = corr_matrix_abs.stack().max()

print(f"The most correlated pair of features: {max_corr} with correlation = {max_value:.2f}")


"
QUESTION 4 :
Logistic Regression assumes a linear decision boundary. Why?
"

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load Parkinson's dataset
parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features
y = parkinsons.data.targets

# Step 2: Split dataset into training and testing sets
# test_size=0.2 → 20% data for testing, 80% for training
# random_state=42 → ensures reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Check the shapes of the splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


"
QUESTION 5 :
Do you think this assumption holds for the Heart Disease dataset? Why or why not?
"


%pip install ucimlrepo

# Logistic Regression assumes a linear decision boundary
# Example with Heart Disease dataset

# 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# 2. Load Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Features and target
X = heart_disease.data.features
y = heart_disease.data.targets.iloc[:, 0]   # make target 1D

# 3. Select only 2 features so we can plot the boundary
print("Available columns:", X.columns.tolist())
X_small = X[["age", "chol"]]   # Example: Age and Cholesterol

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Create meshgrid for decision boundary
x_min, x_max = X_small.iloc[:, 0].min() - 1, X_small.iloc[:, 0].max() + 1
y_min, y_max = X_small.iloc[:, 1].min() - 1, X_small.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 7. Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
sns.scatterplot(
    x="age", y="chol", hue=y, data=X_small,
    palette="coolwarm", edgecolor="k"
)
plt.title("Logistic Regression = Linear Decision Boundary (Heart Disease)")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.show()



" 
QUESTION 6 :
If we increased the number of trees (n_estimators) in Random Forest, how might the performance change?
"

"
QUESTION 7 :
Between Logistic Regression and Random Forest, which model performed better? Why might that be?
"

"
QUESTION 8 :
If we had a much larger dataset with noisy features, which model would you expect to generalize better, and why?
"



# Compare Logistic Regression (linear) vs Random Forest (non-linear)
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

# Load Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets.iloc[:, 0]   # make target 1D

# Handle missing values using imputation
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
log_acc = accuracy_score(y_test, y_pred_log)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

# Print results
print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)


