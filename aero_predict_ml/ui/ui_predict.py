import gradio as gr
import os
import datetime
from aero_predict_ml.src.predict import FlightPredictor

css = """
.gradio-container {
    background-image: url('https://plus.unsplash.com/premium_photo-1678727128583-b7bb1b4763b5?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8cGxhbmUlMjBiYWNrZ3JvdW5kfGVufDB8fDB8fHww');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Ovo uklanja belu pozadinu, senke i okvire sa glavnog panela */
.contain, .gradio-container > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container label, .gradio-container span {
    background-color: transparent !important;
    font-weight: bold !important;
    color: #333 !important;
}

.form {
    background: transparent !important;
    border: none !important;
}
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')

predictor = FlightPredictor(MODEL_PATH, ENCODER_PATH)

ROUTE_DATA = {
    "New York (JFK) -> Los Angeles (LAX)": {"origin": "JFK", "dest": "LAX", "dist": 2475},
    "Chicago (ORD) -> Dallas (DFW)": {"origin": "ORD", "dest": "DFW", "dist": 802},
    "Atlanta (ATL) -> Orlando (MCO)": {"origin": "ATL", "dest": "MCO", "dist": 406},
    "San Francisco (SFO) -> Seattle (SEA)": {"origin": "SFO", "dest": "SEA", "dist": 679},
    "Denver (DEN) -> Las Vegas (LAS)": {"origin": "DEN", "dest": "LAS", "dist": 607}
}

def predict_flight_delay(airline_code, departure_time, route_selection):

    try:
        if not departure_time or len(departure_time) != 4:
            return "Error: Please enter time in HHMM format (e.g., 0815 for 8:15 AM)."

        now = datetime.datetime.now()
        route_info = ROUTE_DATA.get(route_selection)

        sample = {
            'MONTH': now.month,
            'DAY': now.day,
            'DAY_OF_WEEK': now.isoweekday(),
            'AIRLINE': airline_code.strip().upper(),
            'ORIGIN_AIRPORT': route_info['origin'],
            'DESTINATION_AIRPORT': route_info['dest'],
            'SCHEDULED_DEPARTURE': int(departure_time[:2]),
            'DISTANCE': route_info['dist']
        }

        result = predictor.predict(sample)
        probability = result['probability'] * 100

        return f"Prediction for {route_selection}: There is a {probability:.0f}% chance of delay."

    except Exception as e:
        return f"System Error: {str(e)}"

app = gr.Interface(
    fn=predict_flight_delay,
    inputs=[
        gr.Textbox(label="Airline Code", placeholder="e.g., UA, AA, DL"),
        gr.Textbox(label="Scheduled Departure (HHMM)", placeholder="e.g., 1430"),
        gr.Dropdown(choices=list(ROUTE_DATA.keys()), label="Select US Route", value=list(ROUTE_DATA.keys())[0])
    ],
    outputs="text",
    title="AeroPredict-ML: Aviation Analytics System",
    description="ML tool predicting flight delays using Random Forest and real-time parameters.",
    css=css
)

if __name__ == "__main__":
    app.launch()