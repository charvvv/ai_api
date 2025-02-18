from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

# Load models
model1 = tf.keras.models.load_model('models/history_Stroke.h5')
model2 = tf.keras.models.load_model('models/history_Cardio.h5')
model3 = tf.keras.models.load_model('models/history_CFC.h5')

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Health Diagnosis API is running!"

def interpret_stroke_prediction(prediction):
    if prediction[0][0] == 0.0:
        return "Based on your information, you are not at risk for a stroke."
    else:
        return "You may be at risk for a stroke. Please consult a healthcare provider."

def interpret_cardio_prediction(prediction):
    if prediction[0][0] < 1e-10:  # Very small value, close to zero
        return "You have an extremely low risk for cardiovascular issues."
    else:
        return "You may be at risk for cardiovascular issues. Consider visiting a doctor."

def interpret_cfc_prediction(prediction):
    conditions = [
        "Condition 1",  # Replace with your actual condition names
        "Condition 2",
        "Condition 3",
        "Condition 4"  # This would correspond to the last category
    ]
    
    max_prob_index = np.argmax(prediction[0])  # Get the index of the max probability
    if prediction[0][max_prob_index] >= 0.9:  # High confidence (e.g., 0.9 or more)
        return f"Based on your symptoms, it is highly likely that you are affected by {conditions[max_prob_index]}. Please seek medical attention."
    else:
        return "Based on your symptoms, it is unlikely that you are affected by a serious condition, but please monitor your health and consult a healthcare provider if symptoms persist."

def predict_medical(data):
    try:
        # Define the feature sets for each model
        stroke_features = ['age', 'gender', 'hypertension', 'heart_disease', 'smoking', 'avg_glucose_level', 'bmi']
        cardio_features = ['Exercise', 'Depression', 'Diabetes', 'Sex', 'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History', 'Alcohol_Consumption'] # Corrected cardio features
        cfc_features = ['COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT', 'RUNNY_NOSE', 'STUFFY_NOSE', 'FEVER', 'NAUSEA', 'VOMITING', 'DIARRHEA',
                        'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING', 'LOSS_OF_TASTE', 'LOSS_OF_SMELL', 'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR',
                        'SNEEZING', 'PINK_EYE'] # Corrected cfc features

        # Ensure the feature set is matched correctly for each model:
        stroke_data = np.array([[data.get(feature, 0) for feature in stroke_features]])
        cardio_data = np.array([[data.get(feature, 0) for feature in cardio_features]])
        cfc_data = np.array([[data.get(feature, 0) for feature in cfc_features]])

        # Check for missing or invalid data
        if np.any(np.isnan(stroke_data)) or np.any(np.isnan(cardio_data)) or np.any(np.isnan(cfc_data)):
            return jsonify({"error": "Invalid input data: contains null or NaN values."})

        # Predict using the models
        stroke_pred = model1.predict(stroke_data)
        cardio_pred = model2.predict(cardio_data)
        cfc_pred = model3.predict(cfc_data)

        # Return human-readable sentences
        response = {
            "stroke_prediction": interpret_stroke_prediction(stroke_pred),
            "cardio_prediction": interpret_cardio_prediction(cardio_pred),
            "cfc_prediction": interpret_cfc_prediction(cfc_pred)
        }

        return response

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve JSON data from the request
    data = request.json
    print(f"Received data: {data}")

    # Get predictions using the medical data
    response = predict_medical(data)
    return jsonify(response)


    
    # Uncomment this line to start the Flask app when running in production
    app.run(host='0.0.0.0', port=80)