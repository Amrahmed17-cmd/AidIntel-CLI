# AidIntel-CLI

**Medical AI Diagnostic System | Python, Machine Learning, Medical Diagnostics**

A command-line medical diagnostic system that leverages machine learning to provide intelligent health risk assessments. AidIntel-CLI offers AI-powered predictions for multiple medical conditions through an interactive, user-friendly interface.

## 🚀 Features

- **Multi-Condition Diagnostics**: Assess risk for 4 medical conditions:
  - Anemia Risk Assessment
  - Heart Disease Risk Assessment  
  - Flu vs Cold Symptom Checker
  - Osteoporosis Risk Assessment

- **Interactive CLI Interface**: User-friendly command-line interface with guided input collection
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Color-coded Results**: Visual feedback with red/green indicators for easy interpretation
- **Multiple ML Models**: Utilizes KNN, Random Forest, and other algorithms optimized for each condition
- **Comprehensive Input Summary**: Detailed breakdown of user inputs and assessment results

## 📋 Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib (for model evaluation plots)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amrahmed17-cmd/AidIntel-CLI.git
   cd AidIntel-CLI
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **Run the application**
   ```bash
   cd Python
   python AidIntel_CLI.py
   ```

## 🎯 Usage

Launch the application and select from the main menu:

```
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
```

### Example: Anemia Risk Assessment
1. Select option `1` from the main menu
2. Follow the prompts to input your health metrics
3. Receive instant risk assessment with probability scores
4. View comprehensive input summary

### Example Output
```
=== Results ===
Diagnosis: LOW RISK
Your Input Summary:
Hemoglobin: 14.2
Age: 25
Gender: Female
```

## 📁 Project Structure

```
AidIntel-CLI/
├── Python/
│   ├── AidIntel_CLI.py      # Main CLI interface
│   ├── Anemia.py            # Anemia prediction module
│   ├── Heart_Disease.py     # Heart disease prediction module
│   ├── Flu_Cold.py          # Flu/Cold symptom checker
│   └── Osteoporosis.py      # Osteoporosis risk assessment
├── datasets/
│   ├── anemia_dataset.csv
│   ├── heart_disease_dataset.csv
│   ├── flu_cold_dataset.csv
│   └── osteoporosis_dataset.csv
├── models/
│   ├── anemia.py
│   ├── heart_disease.py
│   ├── flu_cold.py
│   └── osteoporosis.py
└── predictors/
```

## 🧠 Machine Learning Models

| Condition | Algorithm | Dataset Size | Key Features |
|-----------|-----------|--------------|--------------|
| Anemia | K-Nearest Neighbors (KNN) | 1,423 records | Hemoglobin, Age, Gender |
| Heart Disease | Random Forest | 1,027 records | Chest Pain, Blood Pressure, Cholesterol |
| Flu/Cold | Classification Model | 502 records | Symptom Severity (1-5 scale) |
| Osteoporosis | KNN with Accuracy Optimization | 1,960 records | Age, BMI, Hormonal Factors |

## 📊 Model Performance

- **Anemia Predictor**: Optimized KNN with k=5
- **Heart Disease**: Random Forest with probability outputs
- **Flu/Cold Checker**: Symptom-based classification with confidence scores
- **Osteoporosis**: Dynamic k-value optimization with accuracy plotting

## 🔧 Technical Details

### Data Processing
- Automated data preprocessing and cleaning
- Feature scaling and normalization
- Categorical variable encoding

### Model Training
- Real-time model training on application start
- Cross-validation for optimal hyperparameters
- Model evaluation with accuracy metrics

### User Input Validation
- Type checking and range validation
- Error handling with user-friendly messages
- Guided input collection with clear prompts


## ⚠️ Disclaimer

**Important**: This tool is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/Amrahmed17-cmd)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/amrahmed17)

## 🙏 Acknowledgments

- Medical datasets used for training models
- Scikit-learn community for ML algorithms
- Open source contributors and medical AI research community

---

⭐ If you found this project helpful, please give it a star on GitHub! 
