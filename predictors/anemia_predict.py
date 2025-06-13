def predict_anemia_from_user_input(model):
    """
    Takes user input for anemia risk factors and returns a prediction.

    Args:
        model: Trained KNeighborsClassifier

    Returns:
        Dictionary with prediction results
    """
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

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Preprocess gender (same as training)
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})

    # Ensure correct column order (important for KNN)
    input_df = input_df[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']]

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None

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
        'probability': float(probability) if probability is not None else None,
        'input_data': user_data
    }


# Example usage
if __name__ == "__main__":
    # Load your trained model
    try:
        model = joblib.load('anemia_model.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        exit()

    # Get prediction
    results = predict_anemia_from_user_input(model)

    # Display results
    print("\n=== Results ===")
    print(f"Diagnosis: \033[1;{31 if results['prediction'] == 1 else 32}m{results['status']}\033[0m")
    if results['probability'] is not None:
        print(f"Confidence: {results['probability']:.1%}")
    print("\nYour Input Summary:")
    for k, v in results['input_data'].items():
        print(f"{k}: {v}")