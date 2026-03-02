from src.preprocessing import preprocess

X, y, scaler = preprocess()

print("Shape of X:", X.shape)
print("Success 🎉")