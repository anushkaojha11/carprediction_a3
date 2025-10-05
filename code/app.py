# app.py
import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from model import predict_selling_price, brand_classes  # <-- Import directly

# -------------------------------
# Default values
# -------------------------------
default_values = {
    "brand": brand_classes[0],
    "year": 2017,
    "max_power": 100,
    "mileage": 10
}

# -------------------------------
# Helpers
# -------------------------------
def create_input_card(label, component):
    return dbc.Card(
        dbc.CardBody([html.H5(label, style={"text-align": "center"}), component]),
        style={"marginBottom": "20px"}
    )

# -------------------------------
# Input cards
# -------------------------------
card_brand = create_input_card(
    "Select Car Brand",
    dcc.Dropdown(
        id="brand",
        options=[{"label": b, "value": b} for b in brand_classes],
        value=default_values["brand"],
    )
)

card_year = create_input_card(
    "Select Year of Manufacture",
    dcc.Input(id="year", type="number", value=default_values["year"], style={"width": "100%"})
)

card_power = create_input_card(
    "Maximum Power (bhp)",
    dcc.Input(id="max_power", type="number", value=default_values["max_power"], style={"width": "100%"})
)

card_mileage = create_input_card(
    "Mileage (km/l)",
    dcc.Input(id="mileage", type="number", value=default_values["mileage"], style={"width": "100%"})
)

card_predicted_price = dbc.Card(
    dbc.CardBody([
        html.H3("Predicted Price Class:", style={"color": "#00AAFF"}),
        html.H2(" ", id="predicted_class", style={"fontWeight": "bold"}),
        html.H4(" ", id="predicted_class_number", style={"fontWeight": "bold", "color": "gray"}),
    ]),
    style={"marginBottom": "20px"}
)

# -------------------------------
# App setup
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.H1("Car Price Prediction", style={"marginBottom": "30px"}),

        dbc.Row([dbc.Col(card_brand, md=6), dbc.Col(card_year, md=6)]),
        dbc.Row([dbc.Col(card_power, md=6), dbc.Col(card_mileage, md=6)]),
        html.Br(),

        dbc.Button("Predict Price", id="submit", color="success", className="me-2"),
        dbc.Button("Clear", id="clear", color="secondary", className="ms-2"),
        html.Br(), html.Br(),

        dbc.Row([dbc.Col(card_predicted_price, md=12)]),
    ],
    fluid=True,
    className="p-4"
)

# -------------------------------
# Callback
# -------------------------------
@app.callback(
    [Output("predicted_class", "children"),
     Output("predicted_class_number", "children"),
     Output("brand", "value"),
     Output("year", "value"),
     Output("max_power", "value"),
     Output("mileage", "value")],
    [Input("submit", "n_clicks"),
     Input("clear", "n_clicks")],
    [State("brand", "value"),
     State("year", "value"),
     State("max_power", "value"),
     State("mileage", "value")],
    prevent_initial_call=True
)
def handle_buttons(submit_clicks, clear_clicks, brand, year, max_power, mileage):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "clear":
        return "", "", default_values["brand"], default_values["year"], default_values["max_power"], default_values["mileage"]
    elif triggered_id == "submit":
        raw_pred, predicted_class = predict_selling_price(
            year=year, max_power=max_power, mileage=mileage, brand=brand
        )
        return predicted_class, f"Class: {int(raw_pred)}", brand, year, max_power, mileage

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    # optional: warm the model so first click is instant
    from model import load_model
    load_model()

    app.run(host="0.0.0.0", port=80, debug=False, use_reloader=False)