import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class AnemiaPredictor:
    def __init__(self):
        self.model = None
        self.required_features = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']

    def load_and_preprocess_data(self, filepath=r'C:\Users\Amr Ahmed\Desktop\AidIntel Terminal\datasets\anemia_dataset.csv'):
        """Load and preprocess the anemia dataset."""
        try:
            # Load data
            df = pd.read_csv(filepath)

            # Display basic info
            print("\n=== Dataset Information ===")
            print(f"Shape: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())
            print("\nMissing values:")
            print(df.isnull().sum())
            print("\nClass distribution:")
            print(df['Result'].value_counts(normalize=True))

            # Feature and target separation
            X = df.drop('Result', axis=1)
            y = df['Result']

            # Convert Gender to numeric
            if X['Gender'].dtype == 'object':
                X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

            return X, y

        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def train_model(self, n_neighbors=1):
        """Train the KNN model and evaluate performance."""
        try:
            X, y = self.load_and_preprocess_data()

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)

            print("\n=== Model Evaluation ===")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Anemia', 'Anemia']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

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
        plt.title("Anemia Accuracy vs. K value")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()

    def predict_from_user_input(self):
        """Take user input and predict anemia risk."""
        if not self.model:
            raise ValueError("Model has not been trained yet.")

        print("\n=== Anemia Risk Assessment ===")
        print("Please provide the following health metrics:\n")

        # Collect user input
        user_data = {
            'Gender': input("Gender (Male/Female): ").strip().capitalize(),
            'Hemoglobin': float(input("Hemoglobin level (g/dL): ")),
            'MCH': float(input("Mean Corpuscular Hemoglobin (MCH, pg): ")),
            'MCHC': float(input("Mean Corpuscular Hemoglobin Concentration (MCHC, g/dL): ")),
            'MCV': float(input("Mean Corpuscular Volume (MCV, fL): "))
        }

        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([user_data])
        input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
        input_df = input_df[self.required_features]

        # Make prediction
        prediction = self.model.predict(input_df)[0]

        # Prepare results
        result_categories = {
            0: ("No Anemia", "green"),
            1: ("Anemia Detected", "red")
        }
        status, color = result_categories[prediction]

        return {
            'prediction': prediction,
            'status': status,
            'color': color,
            'input_data': user_data
        }


# Main execution
if __name__ == "__main__":
    try:
        predictor = AnemiaPredictor()

        # Train the model
        print("Training model...")
        predictor.train_model(n_neighbors=5)  # You can change the n_neighbors value

        # Get prediction from user
        print("\nNow let's assess your anemia risk:")
        results = predictor.predict_from_user_input()

        # Display results
        print("\n=== Results ===")
        color_code = 31 if results['prediction'] == 1 else 32  # Red for Anemia, green for No Anemia
        print(f"Diagnosis: \033[1;{color_code}m{results['status']}\033[0m")

        print("\nYour Input Summary:")
        for k, v in results['input_data'].items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure 'anemia_dataset.csv' exists in the same directory")
        print("2. Check that the file contains the required columns")
        print("3. Verify the file has valid data (Gender, Hemoglobin, MCH, MCHC, MCV, Result)")