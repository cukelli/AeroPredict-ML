import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score


def train_and_compare_models(data_path, model_save_path):
    print("Loading processed dataset...")
    df = pd.read_csv(data_path)

    x = df.drop('ARRIVAL_DELAY', axis=1)
    y = df['ARRIVAL_DELAY']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = {}
    best_mae = float('inf')
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        print(f"Training model: {name}...")
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "R2": r2}
        print(f"-> {name} finished. MAE: {mae:.2f} minutes | R²: {r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = name

    print(f"{'Algorithm':<30} | {'MAE (Error in min)':<20} | {'R² Score':<10}")
    print("-" * 68)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['MAE']:<20.2f} | {metrics['R2']:<10.4f}")

    print(f"\nMost efficient model is: {best_model_name} (MAE: {best_mae:.2f} min)")
    joblib.dump(best_model, model_save_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    data_file = os.path.join(base_dir, 'data', 'processed_flights.csv')
    model_file = os.path.join(base_dir, 'models', 'model.pkl')

    models_dir = os.path.dirname(model_file)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if os.path.exists(data_file):
        train_and_compare_models(data_file, model_file)
    else:
        print(f"ERROR: Processed data file not found at {data_file}")
        print("Please run preprocess.py first.")