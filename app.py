import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

preprocessor = joblib.load("model/preprocessor.joblib")
model = joblib.load("model/logreg_model.joblib")
meta = json.load(open("model/meta.json"))

FEATURES = meta["features"]
NUM_COLS = meta["num_cols"]
THRESHOLD = 0.5

LABELS = {
    "hotel": "Tipe hotel",
    "market_segment": "Segmen pasar",
    "deposit_type": "Tipe deposit",
    "customer_type": "Tipe pelanggan",
}

DEFAULTS = {
    "hotel": "City Hotel", "lead_time": 60, "adults": 2, "children": 0,
    "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
    "market_segment": "Online TA", "deposit_type": "No Deposit",
    "customer_type": "Transient", "is_repeated_guest": 0,
    "previous_cancellations": 0, "booking_changes": 0,
    "required_car_parking_spaces": 0, "total_of_special_requests": 0, "adr": 100,
}


def predict(form):
    row = {}
    for col in FEATURES:
        value = form.get(col, DEFAULTS[col])
        row[col] = float(value) if col in NUM_COLS else value
    X = pd.DataFrame([row], columns=FEATURES)
    proba = float(model.predict_proba(preprocessor.transform(X))[0, 1])
    return proba


@app.route("/", methods=["GET", "POST"])
def index():
    values = dict(DEFAULTS)
    result = None
    error = None
    if request.method == "POST":
        values.update(request.form.to_dict())
        try:
            proba = predict(request.form)
            risky = proba >= THRESHOLD
            result = {
                "label": "Berisiko Batal" if risky else "Tidak Berisiko",
                "risky": risky,
                "proba": round(proba * 100, 1),
                "proba_ok": round((1 - proba) * 100, 1),
            }
        except (ValueError, TypeError):
            error = "Input tidak valid. Pastikan semua kolom angka terisi dengan benar."
    return render_template(
        "index.html", cats=meta["categories"], labels=LABELS,
        values=values, result=result, error=error, metrics=meta["metrics"],
    )


if __name__ == "__main__":
    app.run(debug=True)
