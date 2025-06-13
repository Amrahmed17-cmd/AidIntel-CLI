import sys
from Anemia import AnemiaPredictor
from Heart_Disease import train_model as train_heart, predict_heart_disease
from Flu_Cold import FluColdPredictor
from Osteoporosis import OsteoporosisPredictor


def main_menu():
    while True:
        print("""
========================================
      AidIntel CLI SYSTEM
========================================
Please select a model to use:
  1. Anemia Risk Assessment
  2. Heart Disease Risk Assessment
  3. Flu vs Cold Symptom Checker
  4. Osteoporosis Risk Assessment
  0. Exit
========================================
        """)
        choice = input("Enter your choice (0-4): ").strip()
        if choice == '1':
            print("\n--- Anemia Risk Assessment ---")
            try:
                predictor = AnemiaPredictor()
                predictor.train_model(n_neighbors=5)
                print("\nNow let's assess your anemia risk:")
                results = predictor.predict_from_user_input()
                color_code = 31 if results['prediction'] == 1 else 32
                print(f"\n=== Results ===\nDiagnosis: \033[1;{color_code}m{results['status']}\033[0m")
                print("\nYour Input Summary:")
                for k, v in results['input_data'].items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"\nError: {str(e)}")
            input("\nPress Enter to return to the main menu...")
        elif choice == '2':
            print("\n--- Heart Disease Risk Assessment ---")
            try:
                model, scaler, feature_columns = train_heart()
                results = predict_heart_disease(model, scaler, feature_columns)
                color_code = 31 if results['prediction'] == 1 else 32
                print("\n=== Results ===")
                print(f"Diagnosis: \033[1;{color_code}m{results['diagnosis']}\033[0m")
                print(f"Disease Probability: {results['probability_disease']:.1%}")
                print(f"No Disease Probability: {results['probability_no_disease']:.1%}")
                print("\nYour Input Summary:")
                for col, value in results['input_data'].items():
                    print(f"{col}: {value}")
            except Exception as e:
                print(f"\nError: {str(e)}")
            input("\nPress Enter to return to the main menu...")
        elif choice == '3':
            print("\n--- Flu vs Cold Symptom Checker ---")
            try:
                predictor = FluColdPredictor()
                predictor.load_and_preprocess_data()
                predictor.train_model()
                print("\nNow let's check your symptoms:")
                results = predictor.predict_from_user_input()
                color_code = 31 if results['diagnosis'] == 'FLU' else 32
                print("\n=== Diagnosis ===")
                print(f"Prediction: \033[1;{color_code}m{results['diagnosis']}\033[0m")
                print(f"FLU probability: {results['probability_flu']:.1%}")
                print(f"COLD probability: {results['probability_cold']:.1%}")
                print("\nYour Symptom Summary:")
                for symptom, rating in results['symptoms'].items():
                    print(f"{symptom.replace('_', ' ').title()}: {rating}/5")
            except Exception as e:
                print(f"\nError: {str(e)}")
            input("\nPress Enter to return to the main menu...")
        elif choice == '4':
            print("\n--- Osteoporosis Risk Assessment ---")
            try:
                predictor = OsteoporosisPredictor()
                predictor.load_and_preprocess_data()
                predictor.train_model()
                predictor.evaluate_model()
                predictor.plot_accuracy_vs_k()
                print("\nRunning risk assessment...")
                results = predictor.predict_from_user_input()
                print("\n=== Results ===")
                print(f"Risk Status: {results['status']}")
                print("\nInput Summary:")
                for k, v in results['input_data'].items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"\nError: {str(e)}")
            input("\nPress Enter to return to the main menu...")
        elif choice == '0':
            print("\nThank you for using AidIntel CLI System. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter a number from 0 to 4.")
            input("\nPress Enter to try again...")

if __name__ == "__main__":
    main_menu()