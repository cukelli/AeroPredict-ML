import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def analyze_learning_curve(data_path, plot_save_path):
    df = pd.read_csv(data_path)

    X = df.drop('ARRIVAL_DELAY', axis=1)
    y = df['ARRIVAL_DELAY']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_percentages = [0.1, 0.3, 0.5, 0.75, 1.0]

    training_sizes = []
    test_errors = []

    print("\nStarting learning curve evaluation...")
    print(f"{'Data Used (%)':<15} | {'Num of Flights':<15} | {'MAE (Test Error)':<15}")
    print("-" * 45)

    for percent in train_percentages:
        subset_size = int(len(X_train) * percent)

        x_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]

        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(x_train_subset, y_train_subset)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        training_sizes.append(subset_size)
        test_errors.append(mae)

        print(f"{int(percent * 100):<14}% | {subset_size:<15} | {mae:<15.2f}")

    print("=" * 45)

    print("Generating and saving the learning curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, test_errors, marker='o', linewidth=2, color='b', label='Test Error (MAE)')

    plt.title('Learning Curve: impact of training data size on accuracy for Random Forrest Algorithm')
    plt.xlabel('Number of Training Flights')
    plt.ylabel('Mean Absolute Error (Minutes)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {plot_save_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    data_file = os.path.join(base_dir, 'data', 'processed_flights.csv')

    reports_dir = os.path.join(base_dir, 'reports')

    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    plot_file = os.path.join(reports_dir, 'learning_curve_plot.png')

    if os.path.exists(data_file):
        analyze_learning_curve(data_file, plot_file)
    else:
        print(f"ERROR: Processed data file not found at {data_file}")
