import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


class FluColdPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None  # will hold feature columns used for training

    def load_and_preprocess_data(self, filepath=r'C:\Users\Amr Ahmed\Desktop\AidIntel Terminal\datasets\flu_cold_dataset.csv'):
        """Load and preprocess the flu/cold dataset."""
        try:
            df = pd.read_csv(filepath)

            # Remove invalid values and filter for FLU and COLD
            invalid_values = [2.0, -1.0, 99.0]
            df_clean = df[~df.isin(invalid_values).any(axis=1)]
            df_filtered = df_clean[df_clean['TYPE'].isin(['FLU', 'COLD'])]

            # Remove allergy/covid symptoms (if present)
            allergy_covid_symptoms = [
                'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH',
                'ITCHY_INNER_EAR', 'PINK_EYE',
                'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING'
            ]
            df_features = df_filtered.drop(columns=allergy_covid_symptoms, errors='ignore')
            df_cleaned = df_features.dropna()

            # Prepare features and target
            X = df_cleaned.drop(columns=['TYPE'])
            y = df_cleaned['TYPE'].map({'FLU': 1, 'COLD': 0})

            # Save feature columns for later use in prediction
            self.feature_columns = list(X.columns)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            return X_train, X_test, y_train, y_test, df_cleaned

        except Exception as e:
            raise RuntimeError(f"Error loading or processing data: {str(e)}")

    def train_model(self, n_neighbors=7):
        """Train the KNN model."""
        try:
            # Make sure data is loaded first and feature_columns are set
            if self.feature_columns is None:
                raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")

            # Load data for training
            X_train, X_test, y_train, y_test, df_cleaned = self.load_and_preprocess_data()

            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            print("\n=== Model Evaluation ===")
            print("Class distribution:")
            print(df_cleaned['TYPE'].value_counts(normalize=True))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['COLD', 'FLU']))

            # Plot accuracy vs k
            self.plot_accuracy_vs_k(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise RuntimeError(f"Error training model: {str(e)}")

    def plot_accuracy_vs_k(self, X_train, y_train, X_test, y_test, k_range=range(1, 21)):
        """Plot accuracy for different k values."""
        accuracies = []
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            accuracies.append(model.score(X_test, y_test))

        plt.figure(figsize=(10, 5))
        plt.plot(k_range, accuracies, marker='o', linestyle='-', color='green')
        plt.title("Flu-Cold Accuracy vs. K value")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()

    def predict_from_user_input(self):
        """Take user input for symptoms and predict FLU or COLD."""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        if self.feature_columns is None:
            raise ValueError("Feature columns not initialized. Load and preprocess data first.")

        print("\n=== FLU vs COLD Symptom Checker ===")
        print("Rate each symptom from 0 (none) to 5 (severe):\n")

        user_data = {}
        for symptom in self.feature_columns:
            while True:
                try:
                    rating = float(input(f"{symptom.replace('_', ' ').title()}: "))
                    if 0 <= rating <= 5:
                        user_data[symptom] = rating
                        break
                    else:
                        print("Please enter a value between 0 and 5")
                except ValueError:
                    print("Please enter a valid number")

        input_df = pd.DataFrame([user_data])[self.feature_columns]
        input_scaled = self.scaler.transform(input_df)

        prediction = self.model.predict(input_scaled)[0]
        proba = self.model.predict_proba(input_scaled)[0]
        diagnosis = "FLU" if prediction == 1 else "COLD"

        return {
            'diagnosis': diagnosis,
            'probability_flu': float(proba[1]),
            'probability_cold': float(proba[0]),
            'symptoms': user_data
        }


if __name__ == "__main__":
    try:
        predictor = FluColdPredictor()

        print("Loading and preprocessing data...")
        predictor.load_and_preprocess_data()

        print("Training model...")
        predictor.train_model()

        print("\nNow let's check your symptoms:")
        results = predictor.predict_from_user_input()

        print("\n=== Diagnosis ===")
        color_code = 31 if results['diagnosis'] == 'FLU' else 32  # Red for FLU, green for COLD
        print(f"Prediction: \033[1;{color_code}m{results['diagnosis']}\033[0m")
        print(f"FLU probability: {results['probability_flu']:.1%}")
        print(f"COLD probability: {results['probability_cold']:.1%}")

        print("\nYour Symptom Summary:")
        for symptom, rating in results['symptoms'].items():
            print(f"{symptom.replace('_', ' ').title()}: {rating}/5")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure 'flu_cold_dataset.csv' exists in the specified directory")
        print("2. Check that the file contains the required columns")
        print("3. Verify the file has valid data (FLU/COLD types)")
