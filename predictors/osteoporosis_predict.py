import pandas as pd
from sklearn.preprocessing import LabelEncoder


def predict_from_user_input(model, label_encoders):
    """
    Takes user input for osteoporosis risk factors and returns a prediction.

    Args:
        model: Trained KNeighborsClassifier
        label_encoders: Dictionary of label encoders used during training

    Returns:
        Dictionary with prediction results
    """
    print("\n=== Osteoporosis Risk Assessment ===")
    print("Please answer the following questions:\n")

    # Collect user input
    user_data = {
        'Age': int(input("Age: ")),
        'Gender': input("Gender (Male/Female): ").strip().capitalize(),
        'Hormonal Changes': input("Hormonal Changes (Normal/Postmenopausal/Other): ").strip().capitalize(),
        'Family History': input("Family History of Osteoporosis? (Yes/No): ").strip().capitalize(),
        'Body Weight': input("Body Weight (Underweight/Normal/Overweight): ").strip().capitalize(),
        'Calcium Intake': input("Calcium Intake (Low/Adequate/High): ").strip().capitalize(),
        'Vitamin D Intake': input("Vitamin D Intake (Insufficient/Sufficient/High): ").strip().capitalize(),
        'Physical Activity': input("Physical Activity Level (Sedentary/Moderate/Active): ").strip().capitalize(),
        'Medical Conditions': input(
            "Any relevant medical conditions? (None/Rheumatoid Arthritis/Other): ").strip().capitalize(),
        'Medications': input("Taking any medications? (None/Corticosteroids/Other): ").strip().capitalize(),
        'Prior Fractures': input("Prior fractures? (Yes/No): ").strip().capitalize()
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in input_df.columns:
            # Handle unseen categories safely
            input_df[col] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Prepare results
    risk_categories = {
        0: ("Not At Risk", "green"),
        1: ("At Risk", "red")
    }

    status, color = risk_categories[prediction]

    return {
        'prediction': prediction,
        'status': status,
        'color': color,
        'probability': float(probability),
        'input_data': user_data  # Return original input for reference
    }


# Example usage
if __name__ == "__main__":
    # Load your trained model and encoders (replace with actual loading code)
    # model = joblib.load('osteoporosis_model.joblib')
    # label_encoders = joblib.load('label_encoders.joblib')

    # For demonstration (remove in production)
    print("NOTE: This is a demo. Uncomment model loading code for real usage.")
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier()
    label_encoders = {}  # Should contain your actual encoders

    # Get prediction
    results = predict_from_user_input(model, label_encoders)

    # Display results
    print("\n=== Results ===")
    print(f"Risk Status: {results['status']}")
    print(f"Confidence Score: {results['probability']:.1%}")
    print("\nInput Summary:")
    for k, v in results['input_data'].items():
        print(f"{k}: {v}")