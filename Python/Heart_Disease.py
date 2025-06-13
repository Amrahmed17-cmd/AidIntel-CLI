import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model():
    df = pd.read_csv(r'C:\Users\Amr Ahmed\Desktop\AidIntel Terminal\datasets\heart_disease_dataset.csv')
    print(df.head())
    print(df['target'].value_counts(normalize=True))
    print(df.columns)

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
    plt.title("Heart-Disease Accuracy vs. K value")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    return knn, scaler, X_train.columns

def predict_heart_disease(model, scaler, feature_columns):
    print("\n=== Heart Disease Risk Assessment ===")
    print("Please provide the following medical information:\n")
    features = {
        'age': ("Age (years)", "30-100"),
        'sex': ("Sex (0 = female, 1 = male)", "0/1"),
        'chest_pain': ("Chest pain type (0-3)",
                       "0: typical angina\n1: atypical angina\n2: non-anginal pain\n3: asymptomatic"),
        'resting_bp': ("Resting blood pressure (mm Hg)", "90-200"),
        'cholestrol': ("Serum cholesterol (mg/dL)", "100-600"),
        'fasting_bs': ("Fasting blood sugar > 120 mg/dL (0 = no, 1 = yes)", "0/1"),
        'resting_ecg': ("Resting ECG results (0-2)",
                        "0: normal\n1: ST-T wave abnormality\n2: left ventricular hypertrophy"),
        'max_hr': ("Maximum heart rate achieved", "60-220"),
        'exercise_angina': ("Exercise induced angina (0 = no, 1 = yes)", "0/1"),
        'oldpeak': ("ST depression induced by exercise", "0.0-6.2"),
        'slope': ("Slope of peak exercise ST segment (0-2)", "0: upsloping\n1: flat\n2: downsloping"),
        'ca': ("Number of major vessels colored by fluoroscopy", "0-3"),
        'thal': ("Thalassemia (1-3)", "1: normal\n2: fixed defect\n3: reversible defect")
    }

    user_data = {}
    for col, (description, valid_range) in features.items():
        while True:
            try:
                value = input(f"{description} ({valid_range}): ")
                if col in ['sex', 'fasting_bs', 'exercise_angina']:
                    value = int(value)
                    if value not in [0, 1]:
                        raise ValueError
                elif col in ['chest_pain', 'resting_ecg', 'slope', 'ca', 'thal']:
                    value = int(value)
                    if col == 'chest_pain' and value not in range(4):
                        raise ValueError
                    elif col == 'resting_ecg' and value not in range(3):
                        raise ValueError
                    elif col == 'slope' and value not in range(3):
                        raise ValueError
                    elif col == 'ca' and value not in range(4):
                        raise ValueError
                    elif col == 'thal' and value not in range(1, 4):
                        raise ValueError
                else:
                    value = float(value)
                user_data[col] = value
                break
            except ValueError:
                print(f"Invalid input. Please enter a valid {description.split(' ')[0]}")

    input_df = pd.DataFrame([user_data])
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    diagnosis = "Heart Disease Detected" if prediction == 0 else "No Heart Disease Detected"
    return {
        'prediction': prediction,
        'diagnosis': diagnosis,
        'probability_disease': float(proba[1]),
        'probability_no_disease': float(proba[0]),
        'input_data': user_data
    }

if __name__ == "__main__":
    model, scaler, feature_columns = train_model()
    results = predict_heart_disease(model, scaler, feature_columns)
    print("\n=== Results ===")
    print(f"Diagnosis: \033[1;{31 if results['prediction'] == 0 else 32}m{results['diagnosis']}\033[0m")
    print(f"Disease Probability: {results['probability_disease']:.1%}")
    print(f"No Disease Probability: {results['probability_no_disease']:.1%}")
    print("\nYour Input Summary:")
    features = {
        'age': ("Age (years)", "30-100"),
        'sex': ("Sex (0 = female, 1 = male)", "0/1"),
        'chest_pain': ("Chest pain type (0-3)",
                       "0: typical angina\n1: atypical angina\n2: non-anginal pain\n3: asymptomatic"),
        'resting_bp': ("Resting blood pressure (mm Hg)", "90-200"),
        'cholestrol': ("Serum cholesterol (mg/dL)", "100-600"),
        'fasting_bs': ("Fasting blood sugar > 120 mg/dL (0 = no, 1 = yes)", "0/1"),
        'resting_ecg': ("Resting ECG results (0-2)",
                        "0: normal\n1: ST-T wave abnormality\n2: left ventricular hypertrophy"),
        'max_hr': ("Maximum heart rate achieved", "60-220"),
        'exercise_angina': ("Exercise induced angina (0 = no, 1 = yes)", "0/1"),
        'oldpeak': ("ST depression induced by exercise", "0.0-6.2"),
        'slope': ("Slope of peak exercise ST segment (0-2)", "0: upsloping\n1: flat\n2: downsloping"),
        'ca': ("Number of major vessels colored by fluoroscopy", "0-3"),
        'thal': ("Thalassemia (1-3)", "1: normal\n2: fixed defect\n3: reversible defect")
    }
    for col, value in results['input_data'].items():
        print(f"{features[col][0]}: {value}")