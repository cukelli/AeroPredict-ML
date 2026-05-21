# AeroPredict-ML: Aviation Analytics System 
A machine learning-based system for predicting flight delays using Python. Developing as a master's thesis project.
This project utilizes historical flight data to train regression models capable of estimating departure delays based on operational parameters such as airline, origin, destination, and scheduled departure time.

# 🚀 Features

- **Data Optimization**: Cleans and filters raw flight data, converting scheduled departures into absolute minutes from midnight.
- **Model Comparison**: Implements and evaluates three distinct algorithmic paradigms:
  - Linear Regression (Parametric Baseline)
  - K-Nearest Neighbors (Lazy Learning)
  - Random Forest Regressor (Ensemble Architecture - turned out as the best model)
- **Evaluation**: Tracks Mean Absolute Error (MAE) and R<sup>2</sup> scores to mathematically prove model capability.
- **Learning Curve Analysis**: Includes subset-evaluation scripts to monitor data volume scaling vs. test error behavior.
- **Gradio UI**: Web dashboard that maps textual inputs (airports and airlines) into coded features for delay predictions.

# 📁 Project Structure

```text
AeroPredict-ML/
├── aero_predict_ml/
│   ├── data/
│   │   ├── flights.csv                  # raw flight dataset
│   │   └── processed_flights.csv        # preprocessed dataset
│   ├── evaluation/
│   │   └── random_forrest_evaluation.py # Evaluation of the best model
│   ├── models/
│   │   ├── encoders.pkl                 # serialized label encoders
│   │   └── model.pkl                    # Random Forest regressor
│   ├── reports/
│   │   └── learning_curve_plot.png      # Visualized learning curve
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── predict.py                   # Script for model prediction
│   │   ├── preprocess.py                # Script for preprocessing data
│   │   └── train.py                     # Model training & comparison matrix
│   └── ui/
│       ├── __init__.py
│       └── ui_predict.py                # Gradio interface
├── pyproject.toml
└── README.md
```

# Script executions 
Run commands below in terminal or run it manually through IDE


## 1. Preprocess the raw data
Run this command in terminal or run it manually through IDE
```bash
python aero_predict_ml/src/preprocess.py
```

## 2. Train the models
```bash
python aero_predict_ml/src/train.py
```

## 3.Analyze data limits for the best model (Random Forest):
```bash
python aero_predict_ml/evaluation/random_forrest_evaluation.py
```

## 4. Test individual prediction in terminal (optional):
```bash
python aero_predict_ml/src/predict.py
```

## 5. Launch the web application
```bash
python aero_predict_ml/ui/ui_predict.py
```



## Model Evaluation (Learning Curve)

To determine if the model would benefit from more historical data, a learning curve analysis was conducted on the best-performing algorithm — the **Random Forest Regressor**. 

The graph below shows that as we add more flights to the training set, the error (MAE) drops and then stabilizes. This flattening means the model has learned everything it can from chosen
features. This means our dataset is large enough for this task.
![Learning Curve - Random Forest Regressor](reports/learning_curve_plot.png)