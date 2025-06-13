def predict_heart_disease(model, scaler):
    """
    Takes user input for heart disease risk factors and returns a prediction.

    Args:
        model: Trained KNeighborsClassifier
        scaler: StandardScaler used during training

    Returns:
        Dictionary with prediction results
    """
    print("\n=== Heart Disease Risk Assessment ===")
    print("Please provide the following medical information:\n")

    # Feature descriptions for user-friendly input
    features = {
        'age': ("Age (years)", "30-100"),
        'sex': ("Sex (0 = female, 1 = male)", "0/1"),
        'cp': ("Chest pain type (0-3)", "0: typical angina\n1: atypical angina\n2: non-anginal pain\n3: asymptomatic"),
        'trestbps': ("Resting blood pressure (mm Hg)", "90-200"),
        'chol': ("Serum cholesterol (mg/dL)", "100-600"),
        'fbs': ("Fasting blood sugar > 120 mg/dL (0 = no, 1 = yes)", "0/1"),
        'restecg': ("Resting ECG results (0-2)",
                    "0: normal\n1: ST-T wave abnormality\n2: left ventricular hypertrophy"),
        'thalach': ("Maximum heart rate achieved", "60-220"),
        'exang': ("Exercise induced angina (0 = no, 1 = yes)", "0/1"),
        'oldpeak': ("ST depression induced by exercise", "0.0-6.2"),
        'slope': ("Slope of peak exercise ST segment (0-2)", "0: upsloping\n1: flat\n2: downsloping"),
        'ca': ("Number of major vessels colored by fluoroscopy", "0-3"),
        'thal': ("Thalassemia (1-3)", "1: normal\n2: fixed defect\n3: reversible defect")
    }

    # Collect user input
    user_data = {}
    for col, (description, valid_range) in features.items():
        while True:
            try:
                value = input(f"{description} ({valid_range}): ")
                # Handle categorical vs numerical inputs
                if col in ['sex', 'fbs', 'exang']:
                    value = int(value)
                    if value not in [0, 1]:
                        raise ValueError
                elif col in ['cp', 'restecg', 'slope', 'ca', 'thal']:
                    value = int(value)
                    if col == 'cp' and value not in range(4):
                        raise ValueError
                    elif col == 'restecg' and value not in range(3):
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

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Ensure correct column order
    input_df = input_df[X_train.columns]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # Prepare results
    diagnosis = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    confidence = proba[1] if prediction == 1 else proba[0]

    return {
        'prediction': prediction,
        'diagnosis': diagnosis,
        'confidence': float(confidence),
        'probability_disease': float(proba[1]),
        'probability_no_disease': float(proba[0]),
        'input_data': user_data
    }


# Example usage
if __name__ == "__main__":
    try:
        # Load model and scaler (add this to your training code)
        # joblib.dump(scaler, 'heart_disease_scaler.pkl')
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_disease_scaler.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        exit()

    # Get prediction
    results = predict_heart_disease(model, scaler)

    # Display results
    print("\n=== Results ===")
    print(f"Diagnosis: \033[1;{31 if results['prediction'] == 1 else 32}m{results['diagnosis']}\033[0m")
    print(f"Confidence: {results['confidence']:.1%}")
    print(f"Disease Probability: {results['probability_disease']:.1%}")
    print(f"No Disease Probability: {results['probability_no_disease']:.1%}")

    print("\nYour Input Summary:")
    for col, value in results['input_data'].items():
        print(f"{features[col][0]}: {value}")