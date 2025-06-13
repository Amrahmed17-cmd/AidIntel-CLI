import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class OsteoporosisPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self, filepath=r"C:\Users\Amr Ahmed\Desktop\AidIntel Terminal\datasets\osteoporosis_dataset.csv"):
        df = pd.read_csv(filepath)
        df = df.dropna()

        columns_to_drop = ['Id', 'Race/Ethnicity', 'Alcohol Consumption', 'Smoking']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

        categorical_columns = df.select_dtypes(include='object').columns
        self.label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop(columns=['Osteoporosis'])
        y = df['Osteoporosis']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return df

    def train_model(self, n_neighbors=9):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.model.predict(self.X_test)

        print("\n=== Model Evaluation ===")
        print(f'Accuracy: {accuracy_score(self.y_test, y_pred):.2f}')
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def plot_accuracy_vs_k(self, k_range=range(1, 21)):
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data must be loaded and preprocessed first.")

        accuracies = []
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.X_train, self.y_train)
            score = model.score(self.X_test, self.y_test)
            accuracies.append(score)

        plt.figure(figsize=(10, 5))
        plt.plot(k_range, accuracies, marker='o', linestyle='-', color='blue')
        plt.title("Osteoporosis Accuracy vs. K Value")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.xticks(k_range)
        plt.tight_layout()  # Good for fixing layout issues
        plt.show()

    def predict_from_user_input(self):
        if not self.model or not self.label_encoders:
            raise ValueError("Model and label encoders must be trained first.")

        print("\n=== Osteoporosis Risk Assessment ===")
        print("Please answer the following questions:\n")

        user_data = {
            'Age': int(input("Age: ")),
            'Gender': input("Gender (Male/Female): ").strip().capitalize(),
            'Hormonal Changes': input("Hormonal Changes (Normal/Postmenopausal/Other): ").strip().capitalize(),
            'Family History': input("Family History of Osteoporosis? (Yes/No): ").strip().capitalize(),
            'Body Weight': input("Body Weight (Underweight/Normal/Overweight): ").strip().capitalize(),
            'Calcium Intake': input("Calcium Intake (Low/Adequate/High): ").strip().capitalize(),
            'Vitamin D Intake': input("Vitamin D Intake (Insufficient/Sufficient/High): ").strip().capitalize(),
            'Physical Activity': input("Physical Activity Level (Sedentary/Moderate/Active): ").strip().capitalize(),
            'Medical Conditions': input("Any relevant medical conditions? (None/Rheumatoid arthritis/Other): ").strip().capitalize(),
            'Medications': input("Taking any medications? (None/Corticosteroids/Other): ").strip().capitalize(),
            'Prior Fractures': input("Prior fractures? (Yes/No): ").strip().capitalize()
        }

        input_df = pd.DataFrame([user_data])

        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        prediction = self.model.predict(input_df)[0]

        risk_categories = {
            0: ("Not At Risk", "green"),
            1: ("At Risk", "red")
        }

        status, color = risk_categories[prediction]

        return {
            'prediction': prediction,
            'status': status,
            'color': color,
            'input_data': user_data
        }


if __name__ == "__main__":
    predictor = OsteoporosisPredictor()

    print("Loading and preprocessing data...")
    predictor.load_and_preprocess_data()

    print("\nTraining model...")
    predictor.train_model()

    print("\nEvaluating model...")
    predictor.evaluate_model()

    try:
        print("\nPlotting accuracy for different K values...")
        predictor.plot_accuracy_vs_k()
    except Exception as e:
        print(f"Could not display plot: {e}")

    print("\nRunning risk assessment...")
    results = predictor.predict_from_user_input()

    print("\n=== Results ===")
    print(f"Risk Status: {results['status']}")
    print("\nInput Summary:")
    for k, v in results['input_data'].items():
        print(f"{k}: {v}")
