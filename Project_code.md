"
IRIS FLOWER CLASSIFICATION USING LOGISTICS REGRESSION AND RANDOM FORESTING 


QUESTION 1 :

From the scatterplot/pairplot above which two features seem most useful for separating species?
"

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ==========================
# Load the Iris dataset
# ==========================
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Map numeric labels to species names
species_map = dict(zip(range(3), iris.target_names))
y = y.map(species_map)

# Combine for easy viewing
df = pd.concat([X, y], axis=1)

# ==========================
# Pairplot visualization
# ==========================
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Dataset Features", y=1.02, fontsize=14)
plt.show()

print("Observation from pairplot:")
print("------------------------------------------------")
print("1. Petal length and petal width clearly separate species best.")
print("   - Setosa forms a distinct cluster (small petals, narrow width).")
print("   - Versicolor and Virginica are somewhat overlapping, but still separable.")
print("2. Sepal length and sepal width show significant overlap, making them less useful.")
print("------------------------------------------------\n")

# ==========================
# Decision boundary (using petal length & width)
# ==========================
X_pw = X[["petal length (cm)", "petal width (cm)"]]
X_train, X_test, y_train, y_test = train_test_split(X_pw, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression model
model = LogisticRegression(multi_class="ovr", max_iter=200)
model.fit(X_train, y_train)

# Create decision boundary
x_min, x_max = X_pw.iloc[:, 0].min() - 0.5, X_pw.iloc[:, 0].max() + 0.5
y_min, y_max = X_pw.iloc[:, 1].min() - 0.5, X_pw.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Map species names back to numeric labels for plotting the decision boundary
inverse_species_map = {v: k for k, v in species_map.items()}
Z = np.array([inverse_species_map[name] for name in Z.ravel()]).reshape(xx.shape)

# Debugging prints
print("inverse_species_map:", inverse_species_map)
print("Unique values in Z after mapping:", np.unique(Z))


# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap="Set2")
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="species", data=df, s=60, edgecolor="k")
plt.title("Decision Boundary using Petal Length & Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.show()

# ==========================
# Final Answer
# ==========================
print("Final Answer: The two most useful features for separating species are 'Petal Length' and 'Petal Width'.")


