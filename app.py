import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import json
import pickle

# Configuration
API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="BioAI Insight Engine - Diagnostics Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0c1220; color: #f8fafc; }
    .stSidebar { background-color: #090e1a !important; }
    h1, h2, h3 { font-family: 'Outfit', sans-serif; }
    .badge { padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-high { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-low { background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }
</style>
""", unsafe_allow_html=True)

st.title("🔬 BioAI Insight Engine")
st.write("AI-Powered Cancer Risk Prediction & Literature Search RAG System")

# Check API Health
api_online = False
api_perf = {}
try:
    r = requests.get(f"{API_URL}/health", timeout=2.0)
    if r.status_code == 200:
        api_online = True
        api_data = r.json()
        api_perf = api_data.get("model_performance", {})
except Exception:
    pass

if api_online:
    st.sidebar.success("🟢 Connected to Production API Server")
else:
    st.sidebar.warning("⚠️ FastAPI Server Offline. Running local fallback mode.")

# Define clinical features
feature_names = [
    "Cl.thickness",
    "Cell.size",
    "Cell.shape",
    "Marg.adhesion",
    "Epith.c.size",
    "Bare.nuclei",
    "Bl.cromatin",
    "Normal.nucleoli",
    "Mitoses"
]

# Side bar patient inputs
st.sidebar.header("Patient Feature Inputs")
slider_values = {}
for feat in feature_names:
    slider_values[feat] = st.sidebar.slider(
        feat,
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=1.0
    )

# Prepare prediction request
payload = {
    "Cl.thickness": float(slider_values["Cl.thickness"]),
    "Cell.size": float(slider_values["Cell.size"]),
    "Cell.shape": float(slider_values["Cell.shape"]),
    "Marg.adhesion": float(slider_values["Marg.adhesion"]),
    "Epith.c.size": float(slider_values["Epith.c.size"]),
    "Bare.nuclei": float(slider_values["Bare.nuclei"]),
    "Bl.cromatin": float(slider_values["Bl.cromatin"]),
    "Normal.nucleoli": float(slider_values["Normal.nucleoli"]),
    "Mitoses": float(slider_values["Mitoses"])
}

# 1. PREDICTION SECTION
st.header("Patient Diagnostics")

col1, col2 = st.columns([1, 1])

if api_online:
    # Query API for prediction
    try:
        pred_response = requests.post(f"{API_URL}/predict", json=payload).json()
        
        with col1:
            st.subheader("Classification Outcome")
            pred_label = pred_response["prediction_label"]
            confidence = pred_response["confidence_score"] * 100
            
            if pred_response["prediction"] == 1:
                st.markdown(f"### <span class='badge badge-high'>⚠️ Malignant: {pred_label}</span>", unsafe_allow_html=True)
                st.error(f"High risk detected. Calibrated confidence: {confidence:.2f}%")
            else:
                st.markdown(f"### <span class='badge badge-low'>✅ Benign: {pred_label}</span>", unsafe_allow_html=True)
                st.success(f"Low risk detected. Calibrated confidence: {(100-confidence):.2f}% (Benign)")
                
            # Model probability breakdown table
            probs = pred_response["model_probabilities"]
            st.write("#### Calibrated Model Probability Outputs:")
            st.dataframe(pd.DataFrame({
                "Model": ["Random Forest (Primary)", "Logistic Regression", "Support Vector Machine"],
                "Malignancy Probability": [f"{probs['random_forest']*100:.2f}%", f"{probs['logistic_regression']*100:.2f}%", f"{probs['svm']*100:.2f}%"]
            }))
            
        with col2:
            st.subheader("Local SHAP Attribution")
            shap_attrs = pred_response.get("shap_attributions", [])
            if shap_attrs:
                # Plot SHAP local values
                features = [item["feature"] for item in shap_attrs]
                values = [item["shap_value"] for item in shap_attrs]
                colors = ['#ef4444' if v >= 0 else '#10b981' for v in values]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0c1220')
                ax.set_facecolor('#111827')
                
                bars = ax.barh(features, values, color=colors)
                ax.axvline(0, color='white', linewidth=1, linestyle='--')
                ax.set_title("SHAP Feature Contributions", color='white', fontsize=12)
                ax.tick_params(colors='white', labelsize=10)
                ax.xaxis.grid(True, color='rgba(255,255,255,0.05)')
                
                # Format text
                for bar in bars:
                    width = bar.get_width()
                    align = 'left' if width < 0 else 'right'
                    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                            va='center', ha=align, color='white', fontsize=8, fontweight='bold')
                
                st.pyplot(fig)
            else:
                st.info("Local SHAP values not available.")
                
    except Exception as e:
        st.error(f"Error communicating with API for prediction: {e}")
else:
    # Local fallback prediction
    from src.preprocessing import get_scaler
    from src.explainability import explain_instance
    scaler = get_scaler()
    
    rf_model_path = "data/models/random_forest.pkl"
    if scaler is not None and os.path.exists(rf_model_path):
        with open(rf_model_path, "rb") as f:
            model = pickle.load(f)
            
        raw_values = [payload[k] for k in feature_names]
        sample_scaled = scaler.transform(np.array([raw_values]))
        
        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0][1]
        
        with col1:
            st.subheader("Classification Outcome (Local)")
            if pred == 1:
                st.markdown(f"### <span class='badge badge-high'>⚠️ Malignant</span>", unsafe_allow_html=True)
                st.error(f"Confidence score: {prob*100:.2f}%")
            else:
                st.markdown(f"### <span class='badge badge-low'>✅ Benign</span>", unsafe_allow_html=True)
                st.success(f"Confidence score: {(1 - prob)*100:.2f}% (Benign)")
                
        with col2:
            st.subheader("Local SHAP Attribution (Local)")
            try:
                shap_explanation = explain_instance(sample_scaled[0], "random_forest")
                if shap_explanation:
                    values = shap_explanation["shap_values"]
                    colors = ['#ef4444' if v >= 0 else '#10b981' for v in values]
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#0c1220')
                    ax.set_facecolor('#111827')
                    
                    ax.barh(feature_names, values, color=colors)
                    ax.axvline(0, color='white', linewidth=1, linestyle='--')
                    ax.set_title("SHAP Feature Contributions", color='white', fontsize=12)
                    ax.tick_params(colors='white', labelsize=10)
                    ax.xaxis.grid(True, color='rgba(255,255,255,0.05)')
                    st.pyplot(fig)
            except Exception as ex:
                st.info(f"Could not load local SHAP explanation: {ex}")
    else:
        st.warning("Please start the API server or train the models using python -m src.ml_model to build model files.")

# 2. MODEL METRICS SECTION
st.header("Global Model Analytics")
col3, col4 = st.columns([1, 1])

if api_online and api_perf:
    with col3:
        st.subheader("Model Performance Comparison")
        accs = [api_perf[m]["accuracy"]*100 for m in api_perf]
        aucs = [api_perf[m]["roc_auc"]*100 for m in api_perf]
        models = [m.replace("_", " ").title() for m in api_perf]
        
        df_metrics = pd.DataFrame({
            "Model": models,
            "Accuracy (%)": accs,
            "ROC-AUC (%)": aucs
        }).set_index("Model")
        st.bar_chart(df_metrics)
        
    with col4:
        st.subheader("Global Feature Importance (SHAP)")
        try:
            global_shap = requests.get(f"{API_URL}/explain/global?model_name=random_forest").json()
            features = list(global_shap.keys())
            importances = list(global_shap.values())
            
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0c1220')
            ax.set_facecolor('#111827')
            ax.barh(features[::-1], importances[::-1], color='#00f2fe')
            ax.set_title("Mean Absolute SHAP Impact", color='white', fontsize=12)
            ax.tick_params(colors='white', labelsize=10)
            ax.xaxis.grid(True, color='rgba(255,255,255,0.05)')
            st.pyplot(fig)
        except Exception:
            st.info("Global explanations not reachable.")
else:
    with col3:
        st.info("Metrics not loaded. Ensure API is running and trained.")

# 3. SEMANTIC LITERATURE RAG SEARCH
st.header("Literature RAG Search")
search_query = st.text_input("Query the indexed biomedical corpus:", placeholder="Enter query (e.g., 'Convolutional Neural Networks in breast cancer')")

if search_query:
    if api_online:
        try:
            search_res = requests.post(f"{API_URL}/rag/retrieve", json={"query": search_query, "k": 3}).json()
            st.write(f"Retrieved {len(search_res['results'])} matches in {search_res['latency_ms']:.1f}ms:")
            for res in search_res["results"]:
                with st.expander(f"📄 {res['title']} (Match Score: {res['similarity_score']*100:.1f}%)"):
                    st.write(f"**Authors:** {res['authors']} | **Journal:** {res['journal']} ({res['year']})")
                    st.info(res["abstract"])
        except Exception as e:
            st.error(f"Search API error: {e}")
    else:
        # Local fallback search
        from src.rag_engine import retrieve
        try:
            results = retrieve(search_query, k=3)
            for r in results:
                with st.expander(f"📄 {r['title']} (Local Match Score: {r['similarity_score']*100:.1f}%)"):
                    st.write(f"**Authors:** {r['authors']} | **Journal:** {r['journal']} ({r['year']})")
                    st.info(r["abstract"])
        except Exception as e:
            st.error(f"Local RAG error: {e}")