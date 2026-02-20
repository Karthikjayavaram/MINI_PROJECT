import streamlit as st
import torch
import joblib
import numpy as np
from model import DeepNN

st.set_page_config(page_title="Federated Mental Health Prediction", layout="centered")

# ---------------- LOAD SCALER ----------------
scaler = joblib.load("scaler.pkl")

# ---------------- MODEL CONFIG ----------------
INPUT_SIZE = scaler.mean_.shape[0]
NUM_CLASSES = 4

model = DeepNN(INPUT_SIZE, NUM_CLASSES)
model.load_state_dict(torch.load("fedprox_global_model.pth", map_location="cpu"))
model.eval()

# ---------------- UI TITLE ----------------
st.title("Federated MultiModal Mental Health Prediction")
st.markdown("🔐 Uses Federated Learning (FedProx Global Model)")
st.write("Enter feature values below and click Predict.")

# ---------------- FEATURE NAMES ----------------
feature_names = [
    'Depression_Score', 'Anxiety_Score', 'Stress_Score',
    'Sleep_Quality', 'Social_Engagement',
    'Daily_App_Usage_Min', 'Typing_Speed_WPM',
    'Session_Frequency', 'Idle_Time_Min',
    'Facial_Emotion_Variance', 'Eye_Blink_Rate',
    'Smile_Intensity', 'Head_Motion_Index',
    'MFCC_Mean', 'MFCC_Variance',
    'Pitch_Mean', 'Speech_Rate',
    'Heart_Rate_BPM', 'HRV_Index',
    'Skin_Temperature', 'GSR_Level'
]

# ---------------- CLASS LABELS ----------------
class_labels = {
    0: "Healthy",
    1: "Mild Stress",
    2: "Moderate Stress",
    3: "Severe Stress"
}

# ---------------- INPUT FORM ----------------
inputs = []

with st.form("prediction_form"):

    for name in feature_names:
        val = st.number_input(name, value=0.0)
        inputs.append(val)

    submitted = st.form_submit_button("Predict")

# ---------------- PREDICTION ----------------
if submitted:

    data = np.array(inputs, dtype=np.float32).reshape(1, -1)

    # Scale
    data_scaled = scaler.transform(data)

    tensor = torch.tensor(data_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).numpy().flatten()
        pred = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

    # Display result
    st.success(f"Predicted Mental Health Status: **{class_labels[pred]}**")
    st.info(f"Confidence: {confidence:.2f}%")

    # Show probability breakdown
    st.subheader("Class Probabilities:")
    for i in range(NUM_CLASSES):
        st.write(f"{class_labels[i]}: {probabilities[i]*100:.2f}%")