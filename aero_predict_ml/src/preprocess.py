import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def load_and_clean_data(file_path):
    print("Loading dataset ...")
    df = pd.read_csv(file_path, nrows=100000)

    df = df[df['CANCELLED'] == 0]

    hours = df['SCHEDULED_DEPARTURE'] // 100
    minutes = df['SCHEDULED_DEPARTURE'] % 100
    df['SCHEDULED_DEPARTURE'] = (hours * 60) + minutes

    features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
                'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                'SCHEDULED_DEPARTURE', 'DISTANCE', 'ARRIVAL_DELAY']

    df = df[features].dropna()
    return df


def encode_categorical(df):
    categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    encoders_dict = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders_dict[col] = le

    return df, encoders_dict


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(script_path)

    base_dir = os.path.dirname(src_dir)

    raw_data_path = os.path.join(base_dir, 'data', 'flights.csv')
    processed_data_path = os.path.join(base_dir, 'data', 'processed_flights.csv')
    encoder_path = os.path.join(base_dir, 'models', 'encoders.pkl')

    if not os.path.exists(raw_data_path):
        print(f"Error: File not found:\n{raw_data_path}")
    else:
        print(f"Processing file: {raw_data_path}")

        df_cleaned = load_and_clean_data(raw_data_path)
        df_final, encoders = encode_categorical(df_cleaned)

        models_dir = os.path.dirname(encoder_path)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        df_final.to_csv(processed_data_path, index=False)

        print(f"Succesfully processed files!")
        print(f"Procesed data: {processed_data_path}")
        print(f"Saved enocders: {encoder_path}")

        joblib.dump(encoders, encoder_path)

        print("\n" + "=" * 30)
        print("Checking .pkl file:")

        loaded_encoders = joblib.load(encoder_path)

        for column, encoder in loaded_encoders.items():
            category_num = len(encoder.classes_)
            print(f"Column '{column}': Recognized {category_num} different values.")
            print(f"  Example: {encoder.classes_[:5]}")
