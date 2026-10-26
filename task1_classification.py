# Datheon Internship - Task 1

# Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
classification_report,
confusion_matrix
)

# Step 1: Load the dataset

df = pd.read_csv("iris.txt")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nSpecies Count:")
print(df["species"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Separate features and target

X = df.drop("species", axis=1)
y = df["species"]

# Step 3: Split data into 80% training and 20% testing

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Scale the features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create classification models

models = {
"Logistic Regression": LogisticRegression(max_iter=1000),
"Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = []

# Step 6: Train, predict, and evaluate

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted", zero_division=0))
    


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

results.append({
    "Model": name,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})

print(f"\n{'=' * 40}")
print(name)
print("=" * 40)
print(classification_report(y_test, y_pred, zero_division=0))


# Step 7: Compare model performance

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Step 8: Plot pairplot

sns.pairplot(df, hue="species")
plt.suptitle("Iris Dataset - Pairplot", y=1.02)
plt.show()

# Step 9: Plot confusion matrices

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=ax
)

ax.set_title(f"{name} - Confusion Matrix")
ax.set_xlabel("Predicted Species")
ax.set_ylabel("Actual Species")


plt.tight_layout()
plt.show()

# Step 10: Visualize decision boundaries

# Use petal length and petal width for a 2D visualization.

X_2d = df[["petal_length", "petal_width"]]
y_2d = df["species"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
X_2d, y_2d,
test_size=0.2,
random_state=42,
stratify=y_2d
)

scaler_2d = StandardScaler()
X2_train_scaled = scaler_2d.fit_transform(X2_train)
X2_test_scaled = scaler_2d.transform(X2_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, model) in zip(axes, models.items()):
    model_2d = model.__class__(**model.get_params())
    model_2d.fit(X2_train_scaled, y2_train)
x_min, x_max = X2_train_scaled[:, 0].min() - 1, X2_train_scaled[:, 0].max() + 1
y_min, y_max = X2_train_scaled[:, 1].min() - 1, X2_train_scaled[:, 1].max() + 1

xx, yy = __import__("numpy").meshgrid(
    __import__("numpy").arange(x_min, x_max, 0.02),
    __import__("numpy").arange(y_min, y_max, 0.02)
)

grid = __import__("numpy").c_[xx.ravel(), yy.ravel()]
predictions = model_2d.predict(grid)

class_numbers = pd.Categorical(
    predictions, categories=model_2d.classes_
).codes.reshape(xx.shape)

ax.contourf(xx, yy, class_numbers, alpha=0.3)

actual_numbers = pd.Categorical(
    y2_train, categories=model_2d.classes_
).codes

ax.scatter(
    X2_train_scaled[:, 0],
    X2_train_scaled[:, 1],
    c=actual_numbers,
    edgecolors="black"
)

ax.set_title(f"{name} - Decision Boundary")
ax.set_xlabel("Scaled Petal Length")
ax.set_ylabel("Scaled Petal Width")


plt.tight_layout()
plt.show()

print("\nTask 1 completed successfully!")
