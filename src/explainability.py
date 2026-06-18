import os
import pickle
import numpy as np
import shap

MODELS_DIR = "data/models"

def get_shap_explainer(model_name="random_forest"):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    if model_name == "random_forest":
        explainer = shap.TreeExplainer(model)
        return explainer
    return None

def explain_instance(sample_scaled, model_name="random_forest"):
    """
    Computes local SHAP explanation for a single sample (scaled).
    Returns base_value, shap_values.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    if model_name == "random_forest":
        explainer = shap.TreeExplainer(model)
        if len(sample_scaled.shape) == 1:
            sample_scaled = sample_scaled.reshape(1, -1)
            
        shap_vals = explainer.shap_values(sample_scaled)
        
        # handle list (binary/multi-class outputs)
        if isinstance(shap_vals, list):
            # shap_vals[1] corresponds to class 1
            local_shap = shap_vals[1][0]
            base_val = explainer.expected_value[1]
        elif len(shap_vals.shape) == 3:
            local_shap = shap_vals[0, :, 1]
            base_val = explainer.expected_value[1]
        else:
            local_shap = shap_vals[0]
            base_val = explainer.expected_value
            if isinstance(base_val, np.ndarray) and len(base_val) > 1:
                base_val = base_val[1]
                
        return {
            "base_value": float(base_val),
            "shap_values": [float(v) for v in local_shap]
        }
    elif model_name == "logistic_regression":
        # Linear SHAP explainer for Logistic Regression
        # We can simulate or use simpler linear explanations: SHAP values = coef * (x - mean)
        # Let's use shap.LinearExplainer
        # We need a background dataset, or we can use the model directly
        try:
            from src.preprocessing import load_data, get_scaler
            data = load_data()
            X = data.drop("Class", axis=1)
            scaler = get_scaler()
            X_scaled = scaler.transform(X)
            
            explainer = shap.LinearExplainer(model, X_scaled)
            if len(sample_scaled.shape) == 1:
                sample_scaled = sample_scaled.reshape(1, -1)
            shap_vals = explainer.shap_values(sample_scaled)
            
            # LinearExplainer returns array of shape (num_samples, num_features)
            local_shap = shap_vals[0]
            base_val = explainer.expected_value
            
            return {
                "base_value": float(base_val),
                "shap_values": [float(v) for v in local_shap]
            }
        except Exception as e:
            print(f"Error in Logistic Regression SHAP: {e}")
            
    # Default fallback: return zero attributions if explainer fails or model is SVM
    # Since SVM uses RBF kernel, calculating SHAP values dynamically is extremely slow (KernelExplainer).
    # We will return standard linear/random feature attribution or return a message.
    # To keep latency <120ms, we will indicate that RF is the primary explainable model.
    return None

def get_global_shap_importance(model_name="random_forest"):
    """
    Returns the mean absolute SHAP value for each feature.
    """
    from src.preprocessing import load_data, get_scaler
    data = load_data()
    X = data.drop("Class", axis=1)
    
    scaler = get_scaler()
    if scaler is None:
        return None
        
    X_scaled = scaler.transform(X)
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    if model_name == "random_forest":
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)
        
        if isinstance(shap_vals, list):
            class1_shap = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            class1_shap = shap_vals[:, :, 1]
        else:
            class1_shap = shap_vals
            
        mean_abs_shap = np.abs(class1_shap).mean(axis=0)
        
        features = list(X.columns)
        importance = {features[i]: float(mean_abs_shap[i]) for i in range(len(features))}
        
        sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
        return sorted_importance
    elif model_name == "logistic_regression":
        try:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_vals = explainer.shap_values(X_scaled)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            features = list(X.columns)
            importance = {features[i]: float(mean_abs_shap[i]) for i in range(len(features))}
            sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
            return sorted_importance
        except Exception as e:
            print(f"Error in Global Logistic Regression SHAP: {e}")
            
    return None