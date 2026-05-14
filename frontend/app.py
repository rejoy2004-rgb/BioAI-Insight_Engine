"""
frontend/app.py
===============
Streamlit frontend for the BioAI Insight Engine.

Provides three tabs:
  1. Prediction     — Interactive patient risk assessment
  2. Explainability — SHAP feature importance visualisations
  3. Research Assistant — Semantic PDF search via RAG

IMPORTANT: Run 'python src/train.py' before launching this app.

USAGE:
  streamlit run frontend/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.predict import predict, get_model_info
from src.preprocessing import preprocess, get_feature_names, DATA_PATH
from src.rag_engine import RAGEngine

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BioAI Insight Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers & Cache ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_info():
    """Load model metadata once and cache it for the session."""
    try:
        return get_model_info()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Running preprocessing pipeline...")
def load_training_data():
    """
    Cache the preprocessed training/test split.
    Used by the explainability tab to generate SHAP values.
    """
    try:
        return preprocess(save_scaler=False)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading RAG engine (first time may take 30 s)...")
def load_rag_engine():
    """Initialise and cache the RAG engine for the session."""
    engine = RAGEngine()
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "papers", "sample.pdf")
    try:
        engine.load(pdf_path)
    except FileNotFoundError:
        pass  # Handle gracefully in the UI
    return engine


@st.cache_resource(show_spinner="Loading comparison results...")
def load_comparison_results():
    """Load saved model comparison outputs if they exist."""
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    roc_path = os.path.join(outputs_dir, "roc_curves.png")
    cmp_path = os.path.join(outputs_dir, "model_comparison.png")
    return roc_path, cmp_path


FEATURE_NAMES = get_feature_names()
FEATURE_LABELS = {
    "Cl.thickness":     "Clump Thickness",
    "Cell.size":        "Uniformity of Cell Size",
    "Cell.shape":       "Uniformity of Cell Shape",
    "Marg.adhesion":    "Marginal Adhesion",
    "Epith.c.size":     "Single Epithelial Cell Size",
    "Bare.nuclei":      "Bare Nuclei",
    "Bl.cromatin":      "Bland Chromatin",
    "Normal.nucleoli":  "Normal Nucleoli",
    "Mitoses":          "Mitoses",
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.title("BioAI Insight Engine")
    st.caption("Cancer Risk Prediction & Research Assistant")
    st.divider()

    model_info = load_model_info()
    if model_info:
        st.success(f"✅ Model loaded: **{model_info['model_name']}**")
        st.caption(f"Features: {model_info['n_features']}")
    else:
        st.error("❌ No trained model found.\nRun: `python src/train.py`")

    st.divider()
    st.markdown("**Patient Feature Inputs**")
    st.caption("Adjust sliders to simulate a patient profile.")

    feature_values = {}
    for fname in FEATURE_NAMES:
        label = FEATURE_LABELS.get(fname, fname)
        feature_values[fname] = st.slider(
            label,
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key=f"slider_{fname}",
        )

# ─── Main Tabs ───────────────────────────────────────────────────────────────

tab_predict, tab_explain, tab_research = st.tabs([
    "🔬 Prediction",
    "📊 Explainability",
    "📄 Research Assistant",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.header("🔬 Cancer Risk Prediction")
    st.write(
        "Adjust the feature sliders in the sidebar to simulate a patient profile. "
        "The model returns a risk assessment based on the Wisconsin Breast Cancer dataset."
    )

    if not model_info:
        st.warning("Please run `python src/train.py` to train and save the model first.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Feature Profile")
        df_input = pd.DataFrame(
            {"Feature": list(feature_values.keys()), "Value": list(feature_values.values())}
        )
        df_input["Label"] = df_input["Feature"].map(FEATURE_LABELS)
        st.dataframe(
            df_input[["Label", "Value"]].rename(columns={"Label": "Feature Name"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("Prediction Result")

        if st.button("🧬 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                try:
                    features_list = [feature_values[f] for f in FEATURE_NAMES]
                    result = predict(features_list)

                    # Display result prominently
                    if result["prediction"] == 1:
                        st.error(f"⚠️ **{result['label']} Detected**")
                    else:
                        st.success(f"✅ **{result['label']}**")

                    # Confidence gauge
                    conf_pct = result["confidence"] * 100
                    st.metric("Confidence Score", f"{conf_pct:.1f}%")
                    st.progress(result["confidence"])

                    st.caption(f"Model: {result['model_used']}")

                    # Risk interpretation
                    st.info(
                        "**Interpretation:**\n\n"
                        "- Score < 50% → Lower risk profile\n"
                        "- Score 50–80% → Moderate risk, further tests recommended\n"
                        "- Score > 80% → High risk, clinical review strongly advised\n\n"
                        "*This tool is for research purposes only and does not constitute medical advice.*"
                    )

                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Model performance metrics
    st.divider()
    st.subheader("📈 Model Performance")

    roc_path, cmp_path = load_comparison_results()

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves — All Models", use_container_width=True)
        else:
            st.info("ROC curve not found. Run `python src/train.py` to generate it.")

    with col_b:
        if os.path.exists(cmp_path):
            st.image(cmp_path, caption="Model Metrics Comparison", use_container_width=True)
        else:
            st.info("Metrics chart not found. Run `python src/train.py` to generate it.")

    # Confusion matrices
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    cm_files = [f for f in os.listdir(outputs_dir) if f.startswith("cm_")] \
        if os.path.exists(outputs_dir) else []

    if cm_files:
        st.subheader("Confusion Matrices")
        cols = st.columns(len(cm_files))
        for col, fname in zip(cols, cm_files):
            label = fname.replace("cm_", "").replace(".png", "").replace("_", " ").title()
            col.image(os.path.join(outputs_dir, fname), caption=label, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

with tab_explain:
    st.header("📊 Model Explainability (SHAP)")
    st.write(
        "SHAP (SHapley Additive exPlanations) attributes each feature's contribution "
        "to a prediction. This is critical for healthcare AI where decisions must be auditable."
    )

    training_data = load_training_data()

    if training_data is None:
        st.warning("Could not load training data. Check data/cancer.csv exists.")
    elif not model_info:
        st.warning("Train the model first: `python src/train.py`")
    else:
        X_train, X_test, y_train, y_test, scaler, feat_names = training_data

        # Load the fitted model for SHAP
        try:
            import joblib
            MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
            artifacts = joblib.load(MODEL_PATH)
            trained_model = artifacts["model"]

            col_e1, col_e2 = st.columns([1, 1])

            with col_e1:
                st.subheader("Global Feature Importance")
                shap_summary_path = os.path.join(
                    os.path.dirname(__file__), "..", "outputs", "shap_summary.png"
                )

                if not os.path.exists(shap_summary_path):
                    with st.spinner("Computing SHAP values (may take 15–30 s)..."):
                        try:
                            from src.explainability import (
                                compute_shap_values,
                                plot_shap_summary,
                            )
                            shap_vals, _ = compute_shap_values(
                                trained_model, X_train, X_test[:50]
                            )
                            fig = plot_shap_summary(
                                shap_vals, X_test[:50], feat_names, save=True
                            )
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"SHAP computation failed: {e}")
                            st.caption("Install shap: pip install shap")
                else:
                    st.image(shap_summary_path, use_container_width=True)

            with col_e2:
                st.subheader("Current Patient Explanation")
                st.caption("This shows how each feature affects YOUR current slider values.")

                if st.button("🔍 Explain This Patient", use_container_width=True):
                    with st.spinner("Computing individual SHAP values..."):
                        try:
                            from src.explainability import plot_shap_single_prediction
                            sample_raw = np.array([feature_values[f] for f in FEATURE_NAMES])
                            sample_scaled = scaler.transform(sample_raw.reshape(1, -1))
                            fig = plot_shap_single_prediction(
                                trained_model,
                                X_train,
                                sample_scaled,
                                feat_names,
                                save=False,
                            )
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not generate explanation: {e}")

            # Feature importance bar chart (always available, no SHAP needed)
            st.divider()
            st.subheader("Feature Importance (from model)")
            from src.explainability import get_feature_importances_df
            fi_df = get_feature_importances_df(trained_model, feat_names)
            fi_df["Feature Label"] = fi_df["Feature"].map(FEATURE_LABELS).fillna(fi_df["Feature"])

            fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
            ax_fi.barh(fi_df["Feature Label"], fi_df["Importance"], color="#3498db")
            ax_fi.set_xlabel("Importance")
            ax_fi.set_title("Feature Importance")
            ax_fi.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_fi)

        except Exception as e:
            st.error(f"Could not load model for explainability: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESEARCH ASSISTANT (RAG)
# ══════════════════════════════════════════════════════════════════════════════

with tab_research:
    st.header("📄 Research Paper Assistant")
    st.write(
        "Ask questions about the loaded research paper. The RAG engine finds the most "
        "semantically relevant passages using dense vector search (FAISS + SentenceTransformers)."
    )

    rag = load_rag_engine()

    if not rag.is_loaded:
        st.warning(
            "No PDF loaded. Place a research paper at `papers/sample.pdf` and restart the app."
        )
    else:
        st.success(f"✅ Research paper indexed — {len(rag.chunks)} chunks ready for search.")

        # Suggested questions
        st.subheader("Try a question:")
        suggested = [
            "What is the main finding of this research?",
            "What methodology was used?",
            "What are the conclusions?",
            "What dataset was used in this study?",
        ]
        for q in suggested:
            if st.button(q, key=f"sug_{q[:20]}"):
                st.session_state["rag_query"] = q

        # Query input
        query = st.text_input(
            "Or type your own question:",
            value=st.session_state.get("rag_query", ""),
            placeholder="e.g. What methods were used for classification?",
            key="rag_input",
        )

        top_k = st.slider("Number of results", 1, 5, 3)

        if st.button("🔍 Search", type="primary") and query.strip():
            with st.spinner("Searching paper..."):
                try:
                    results = rag.query(query, top_k=top_k)
                    st.subheader(f"Top {len(results)} Relevant Passages")
                    for i, (passage, score) in enumerate(results, 1):
                        with st.expander(f"Result {i}  —  Relevance: {score:.3f}", expanded=(i == 1)):
                            st.write(passage)
                except Exception as e:
                    st.error(f"Search failed: {e}")

        # How RAG works
        with st.expander("ℹ️ How does this work?"):
            st.markdown(
                """
                **Retrieval-Augmented Generation (RAG) Pipeline:**

                1. **PDF Parsing** — PyMuPDF extracts text from every page
                2. **Chunking** — Text is split into overlapping 500-character segments
                3. **Embedding** — Each chunk is encoded by `all-MiniLM-L6-v2` (SentenceTransformer)
                4. **Indexing** — FAISS stores all embeddings in a flat inner-product index
                5. **Retrieval** — Your query is embedded and the nearest chunks are returned

                This is a **retrieval-only** RAG system. A full RAG pipeline would pass the
                retrieved chunks to an LLM (e.g. Claude, GPT-4) to generate a coherent answer.
                """
            )

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "BioAI Insight Engine · Built with Streamlit, scikit-learn, FAISS, SHAP · "
    "For research and educational purposes only."
)
