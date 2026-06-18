import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from src.preprocessing import load_data, SCALER_PATH

MODELS_DIR = "data/models"

def train_pipeline(seed=48):
    # Load raw data
    data = load_data()
    X = data.drop("Class", axis=1)
    y = data["Class"]
    
    # Strict 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    # Initialize models
    # Logistic Regression
    lr = LogisticRegression(random_state=seed, max_iter=1000)
    # SVM (probability=True is required for predict_proba/confidence scores)
    svm = SVC(probability=True, random_state=seed, C=1.0, kernel='rbf')
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
    
    models = {
        "logistic_regression": lr,
        "svm": svm,
        "random_forest": rf
    }
    
    metrics = {}
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        # Classification report dictionary
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()
        
        metrics[name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "confusion_matrix": cm,
            "classification_report": report
        }
        
        # Save model file
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
            
    # Save metrics
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Models and scaler trained and saved successfully!")
    print(f"Logistic Regression: Accuracy={metrics['logistic_regression']['accuracy']:.4f}, AUC={metrics['logistic_regression']['roc_auc']:.4f}")
    print(f"SVM: Accuracy={metrics['svm']['accuracy']:.4f}, AUC={metrics['svm']['roc_auc']:.4f}")
    print(f"Random Forest: Accuracy={metrics['random_forest']['accuracy']:.4f}, AUC={metrics['random_forest']['roc_auc']:.4f}")
    
    return metrics

if __name__ == "__main__":
    train_pipeline()