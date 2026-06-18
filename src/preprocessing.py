import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "data/scaler.pkl"

def load_data():
    data = pd.read_csv("data/cancer.csv")
    
    # Remove Id column (not useful for prediction)
    if "Id" in data.columns:
        data = data.drop(["Id"], axis=1)
        
    # Remove rows with missing values
    data = data.dropna()
    return data

def preprocess():
    data = load_data()

    # Features (everything except Class)
    X = data.drop("Class", axis=1)
    
    # Target variable
    y = data["Class"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled, y, scaler

def get_scaler():
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    return None