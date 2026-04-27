import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_flight_model(data_path, model_save_path):

    df = pd.read_csv(data_path)

    X = df.drop('IS_DELAYED', axis=1)
    y = df['IS_DELAYED']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"TRAINING SUCCESSFUL: Model accuracy is {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_save_path)
    print(f"Model saved successfully at: {model_save_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    data_file = os.path.join(base_dir, 'data', 'processed_flights.csv')
    model_file = os.path.join(base_dir, 'models', 'model.pkl')

    models_dir = os.path.dirname(model_file)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if os.path.exists(data_file):
        train_flight_model(data_file, model_file)
    else:
        print(f"ERROR: Processed data file not found at {data_file}")
        print("Please run preprocess.py first.")