import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
df = pd.read_csv('flu_cold_dataset.csv')

invalid_values = [2.0, -1.0, 99.0]
df_clean = df[~df.isin(invalid_values).any(axis=1)]

df_filtered = df_clean[df_clean['TYPE'].isin(['FLU', 'COLD'])]

allergy_covid_symptoms = [
    'ITCHY_NOSE',
    'ITCHY_EYES',
    'ITCHY_MOUTH',
    'ITCHY_INNER_EAR',
    'PINK_EYE',
    'SHORTNESS_OF_BREATH',
    'DIFFICULTY_BREATHING'
]
df_features = df_filtered.drop(columns=allergy_covid_symptoms)

df_cleaned = df_features.dropna()

X = df_cleaned.drop(columns=['TYPE'])
y = df_cleaned['TYPE'].map({'FLU': 1, 'COLD': 0})  # Encode labels

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Label encoder for inverse mapping in report
le = LabelEncoder()
y_true_decoded = le.fit_transform(df_cleaned['TYPE'])  # Fit on FLU/COLD only
print("Class distribution:")
print(df_cleaned['TYPE'].value_counts(normalize=True))
print()

# Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


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

joblib.dump(knn, 'flu_cold_model.pkl')