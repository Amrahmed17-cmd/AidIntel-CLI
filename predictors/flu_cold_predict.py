def predict_flu_or_cold(model, scaler):
    """
    Takes user input for symptoms and predicts whether it's FLU or COLD.

    Args:
        model: Trained KNeighborsClassifier
        scaler: MinMaxScaler used during training

    Returns:
        Dictionary with prediction results
    """
    print("\n=== FLU vs COLD Symptom Checker ===")
    print("Rate each symptom from 0 (none) to 5 (severe):\n")

    # List of symptoms used in the model (excluding dropped columns)
    symptoms = [
        'RUNNY_NOSE', 'STUFFY_NOSE', 'SORE_THROAT', 'COUGH',
        'CHILLS', 'BODY_ACHES', 'HEADACHE', 'FEVER',
        'NAUSEA', 'VOMITING', 'SNEEZING', 'FATIGUE'
    ]

    # Collect user input
    user_data = {}
    for symptom in symptoms:
        while True:
            try:
                rating = float(input(f"{symptom.replace('_', ' ').title()}: "))
                if 0 <= rating <= 5:
                    user_data[symptom] = rating
                    break
                else:
                    print("Please enter a value between 0 and 5")
            except ValueError:
                print("Please enter a number")

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Ensure correct column order
    input_df = input_df[symptoms]

    # Scale features (same as training)
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # Prepare results
    diagnosis = "FLU" if prediction == 1 else "COLD"
    confidence = proba[1] if prediction == 1 else proba[0]

    return {
        'diagnosis': diagnosis,
        'confidence': float(confidence),
        'probability_flu': float(proba[1]),
        'probability_cold': float(proba[0]),
        'symptoms': user_data
    }


# Example usage
if __name__ == "__main__":
    # Load model and scaler
    try:
        model = joblib.load('flu_cold_model.pkl')
        scaler = joblib.load('flu_cold_scaler.pkl')  # You'll need to save this during training
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        exit()

    # Get prediction
    results = predict_flu_or_cold(model, scaler)

    # Display results
    print("\n=== Diagnosis ===")
    print(f"Prediction: \033[1;{31 if results['diagnosis'] == 'FLU' else 32}m{results['diagnosis']}\033[0m")
    print(f"Confidence: {results['confidence']:.1%}")
    print(f"FLU probability: {results['probability_flu']:.1%}")
    print(f"COLD probability: {results['probability_cold']:.1%}")

    print("\nYour Symptom Summary:")
    for symptom, rating in results['symptoms'].items():
        print(f"{symptom.replace('_', ' ').title()}: {rating}/5")