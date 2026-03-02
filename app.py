import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import preprocess
from sklearn.metrics import accuracy_score

# Title
st.title("BioAI Insight Engine")

# Load data
X, y, scaler = preprocess()

# Train model
model = RandomForestClassifier()
model.fit(X, y)
pred_all = model.predict(X)
accuracy = accuracy_score(y, pred_all)

st.write(f"Model Accuracy: {accuracy*100:.2f}%")

import matplotlib.pyplot as plt

st.subheader("Feature Importance")

importances = model.feature_importances_

fig, ax = plt.subplots()
ax.bar(range(len(importances)), importances)
ax.set_title("Feature Importance")
st.pyplot(fig)

# Sidebar input
st.sidebar.header("Patient Input Features")

values = []
for i in range(X.shape[1]):
    val = st.sidebar.slider(f"Feature {i+1}", -3.0, 3.0, 0.0)
    values.append(val)

# Convert input to array
sample = np.array([values])

# Predict
prob = model.predict_proba(sample)[0][1]

st.subheader("Prediction Result")

st.write(f"Confidence Score: {prob*100:.2f}%")

if prob > 0.5:
    st.error("⚠ High Cancer Risk Detected")
else:
    st.success("Low Cancer Risk")

# Show result
st.subheader("Prediction Result")

if prob > 0.5:
    st.error("⚠ High Cancer Risk Detected")
else:
    st.success("Low Cancer Risk")