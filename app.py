import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import shap
import matplotlib.pyplot as plt

# Define Bayesian NN (same as training)
class BayesianNN(nn.Module):
    def __init__(self, input_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.2, train=True)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Load models
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

input_dim = len(scaler.mean_)
bayesian_model = BayesianNN(input_dim)
bayesian_model.load_state_dict(torch.load("models/bayesian_model.pth"))
bayesian_model.eval()

# Streamlit UI
st.title("HeartGuard: Heart Disease Prediction with Explainability")

# Inputs with no defaults
age = st.number_input("Age", min_value=20, max_value=100)
sex = st.selectbox("Sex", ["Select", "Male", "Female"],index=0)
cp = st.selectbox("Chest Pain Type", ["Select", 1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=400)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Select", 0, 1])
restecg = st.selectbox("Resting ECG", ["Select", 0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=210)
exang = st.selectbox("Exercise Induced Angina", ["Select", 0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0)
slope = st.selectbox("Slope of ST Segment", ["Select", 1, 2, 3])
ca = st.selectbox("Number of Major Vessels", ["Select", 0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Select", 3, 6, 7])

if st.button("Predict Risk"):
    # Validation
    if (
        sex == "Select" or cp == "Select" or fbs == "Select" or
        restecg == "Select" or exang == "Select" or slope == "Select" or
        ca == "Select" or thal == "Select" or age == 0 or trestbps == 0 or
        chol == 0 or thalach == 0 or oldpeak == 0.0
    ):
        st.error("⚠️ Please fill in all fields before predicting.")
    else:
        # Prepare features
        features = np.array([[
            age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal
        ]])

        features_scaled = scaler.transform(features)

        # XGBoost prediction
        xgb_pred = xgb_model.predict_proba(features_scaled)[0][1]

        # Bayesian NN prediction
        tensor_features = torch.tensor(features_scaled, dtype=torch.float32)
        bayesian_pred = bayesian_model(tensor_features).detach().numpy()[0][0]

        # Hybrid probability
        probability = (xgb_pred + bayesian_pred) / 2

        # Verdict
        if probability < 0.3:
            risk = "Low"
            advice = "Maintain healthy lifestyle."
        elif probability < 0.6:
            risk = "Moderate"
            advice = "Regular exercise and diet control."
        elif probability < 0.8:
            risk = "High"
            advice = "Consult cardiologist immediately."
        else:
            risk = "Critical"
            advice = "Urgent medical intervention required."

        st.metric("Probability of Heart Disease", f"{probability:.2f}")
        st.warning(f"Risk Level: {risk}")
        st.info(f"Advice: {advice}")

        # Feature names
        feature_names = [
            "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol",
            "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate Achieved",
            "Exercise Induced Angina", "ST Depression", "Slope of ST Segment",
            "Number of Major Vessels", "Thalassemia"
        ]

        
        
       
        
        # SHAP Explainability
        st.subheader("Feature Contribution (SHAP Values)")

        try:
            # Explain the model's probability output
            def predict_probability(data):
                data = np.asarray(data, dtype=np.float64)
                return xgb_model.predict_proba(data)[:, 1]

            background = np.zeros((1, features_scaled.shape[1]))
            explainer = shap.KernelExplainer(
                predict_probability,
                background
            )

            shap_values = explainer.shap_values(
                features_scaled.astype(np.float64),
                nsamples=100
            )

            values = np.asarray(shap_values, dtype=np.float64)

            if values.ndim == 3:
                values = values[:, :, 1]

            values = values.reshape(-1, len(feature_names))

            st.subheader("Feature Contribution for This Prediction")
            fig, ax = plt.subplots()
            ax.barh(feature_names, values[0])
            ax.set_xlabel("SHAP contribution to model output")
            ax.set_title("Feature Contributions")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")
