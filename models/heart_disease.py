import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("heart_disease_dataset.csv")
print(df.head())

print(df['target'].value_counts(normalize=True))

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

k_range = range(1, 21)
accuracies = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    accuracies.append(score)

plt.figure(figsize=(10,5))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='green')
plt.title("Accuracy vs. K value")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

joblib.dump(knn, 'heart_disease_model.pkl')