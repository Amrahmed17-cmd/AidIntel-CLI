import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess data
df = pd.read_csv("osteoporosis_dataset.csv")
df = df.dropna()

# Remove unnecessary columns
columns_to_drop = ['Id', 'Race/Ethnicity', 'Alcohol Consumption', 'Smoking']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Encode categorical variables
categorical_columns = df.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data
X = df.drop(columns=['Osteoporosis'])
y = df['Osteoporosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

