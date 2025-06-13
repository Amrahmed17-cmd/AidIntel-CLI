import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
print("Loading and exploring data...")
df = pd.read_csv('anemia_dataset.csv')

# Display basic information
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

print(df['Result'].value_counts(normalize=True))

# Feature and target separation
X = df.drop('Result', axis=1)
y = df['Result']

# Data preprocessing
print("\nPreprocessing data...")
# Convert Gender to numeric if not already (1 for Male, 0 for Female)
if X['Gender'].dtype == 'object':
    X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN
print("\nTraining KNN model...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("\nImproved Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

k_range = range(1, 21)
accuracies = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    accuracies.append(score)

plt.figure(figsize=(10,5))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='green')
plt.title("Accuracy vs. K value")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

joblib.dump(model, 'anemia_model.pkl')