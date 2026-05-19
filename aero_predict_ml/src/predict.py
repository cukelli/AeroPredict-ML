import pandas as pd
import joblib
import os


class FlightPredictor:

    def __init__(self, model_path, encoder_path):
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError("Model or Encoder file not found. Run training first.")

        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoder_path)

    def predict(self, input_data):

        df = pd.DataFrame([input_data])

        for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
            if col in df.columns:
                le = self.encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        feature_order = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
                         'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                         'SCHEDULED_DEPARTURE', 'DISTANCE']

        df = df[feature_order]

        delay_minutes = self.model.predict(df)[0]

        return {
            "predicted_delay_minutes": max(0, round(float(delay_minutes)))
        }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
    ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')

    predictor = FlightPredictor(MODEL_PATH, ENCODER_PATH)

    sample_flight = {
        'MONTH': 5,
        'DAY': 15,
        'DAY_OF_WEEK': 3,
        'AIRLINE': 'AA',
        'ORIGIN_AIRPORT': 'JFK',
        'DESTINATION_AIRPORT': 'LAX',
        'SCHEDULED_DEPARTURE': 870,
        'DISTANCE': 2475
    }

    result = predictor.predict(sample_flight)
    print(f"Prediction Result: Estimated delay is {result['predicted_delay_minutes']} minutes.")