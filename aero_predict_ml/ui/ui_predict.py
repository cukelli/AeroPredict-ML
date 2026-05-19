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

.contain, .gradio-container > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container label, 
.gradio-container span, 
.gradio-container h1, 
.gradio-container p,
.gradio-container .markdown-text {
    background-color: transparent !important;
    font-weight: bold !important;
    color: white !important;
    text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.8);
}
.gradio-container h1 {
    background-color: transparent !important;
    font-weight: bold !important;
    color: #000000 !important; 
    text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.6); 
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

        hours = int(departure_time[:2])
        minutes = int(departure_time[2:])
        total_minutes_from_midnight = (hours * 60) + minutes

        sample = {
            'MONTH': now.month,
            'DAY': now.day,
            'DAY_OF_WEEK': now.isoweekday(),
            'AIRLINE': airline_code.strip().upper(),
            'ORIGIN_AIRPORT': route_info['origin'],
            'DESTINATION_AIRPORT': route_info['dest'],
            'SCHEDULED_DEPARTURE': total_minutes_from_midnight,
            'DISTANCE': route_info['dist']
        }

        result = predictor.predict(sample)
        delay_minutes = result['predicted_delay_minutes']

        if delay_minutes == 0:
            return f"Prediction for {route_selection} with {airline_code.upper()}:\nThe flight is expected to be ON " \
                   f"TIME."
        else:
            return f"Prediction for {route_selection} with {airline_code.upper()}:\nEstimated delay is {delay_minutes} minutes."

    except Exception as e:
        return f"System Error: {str(e)}"


app = gr.Interface(
    fn=predict_flight_delay,
    inputs=[
        gr.Textbox(label="Airline Code", placeholder="e.g., UA, AA, DL"),
        gr.Textbox(label="Scheduled Departure (HHMM)", placeholder="e.g., 1430"),
        gr.Dropdown(choices=list(ROUTE_DATA.keys()), label="Select US Route", value=list(ROUTE_DATA.keys())[0])
    ],
    outputs=gr.Textbox(label="Flight delay"),
    title="AeroPredict-ML: Aviation Analytics System",
    css=css
)

if __name__ == "__main__":
    app.launch()
