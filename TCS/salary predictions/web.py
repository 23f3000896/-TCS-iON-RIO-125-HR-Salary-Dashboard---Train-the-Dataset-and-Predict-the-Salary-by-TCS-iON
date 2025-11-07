from flask import Flask, render_template, request
import pickle
import numpy as np

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# ✅ Home page route
@app.route("/")
def home():
    return render_template("home.html")

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert form inputs into float values
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        return render_template("res.html", prediction_text=f"The Salary is Rs {prediction:.2f}")
    except Exception as e:
        return render_template("res.html", prediction_text=f"Error: {e}")

# ✅ Run Flask server
if __name__ == "__main__":
    app.run(debug=True)

