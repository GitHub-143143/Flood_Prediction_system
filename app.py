from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import pickle

app = Flask(__name__)

# -------------------------------
# Load trained model components
# -------------------------------
model = load("model/floods.save")
scaler = load("model/transform.save")
columns = pickle.load(open("model/columns.save", "rb"))

print("Model Loaded Successfully")
print("Expected Feature Order:", columns)


# -------------------------------
# Home Page
# -------------------------------
@app.route('/')
def home():
    return render_template("home.html")


# -------------------------------
# Prediction Form Page
# -------------------------------
@app.route('/predict')
def predict_page():
    return render_template("index.html")


# -------------------------------
# Prediction Logic
# -------------------------------
@app.route('/data_predict', methods=['POST'])
def data_predict():
    try:
        # Collect inputs in exact training order
        input_data = []
        for col in columns:
            if col in request.form:
                input_data.append(float(request.form[col]))
            else:
                return f"Missing input field: {col}"

        # Convert to DataFrame
        data = pd.DataFrame([input_data], columns=columns)

        # Scale input
        data_scaled = scaler.transform(data)

        # Get probability
        proba = model.predict_proba(data_scaled)[0][1]
        print("Flood Probability:", proba)

        # Custom Threshold (important for imbalance)
        threshold = 0.40

        if proba >= threshold:
            return render_template(
                "chance.html",
                probability=round(proba * 100, 2)
            )
        else:
            return render_template(
                "nochance.html",
                probability=round(proba * 100, 2)
            )

    except Exception as e:
        return f"Error occurred: {str(e)}"


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
