import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv("data/cancer.csv")
    
    # Remove Id column (not useful for prediction)
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

    return X_scaled, y, scaler