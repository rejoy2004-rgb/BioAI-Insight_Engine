from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing import preprocess
from sklearn.model_selection import GridSearchCV

# Load processed data
X, y, scaler = preprocess()

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("=== Hyperparameter Tuning (Random Forest) ===")

# Define parameters we want to test
params = {
    'n_estimators': [50, 100, 200],   # number of trees
    'max_depth': [None, 5, 10]        # tree depth
}

# Create GridSearch object
grid = GridSearchCV(
    RandomForestClassifier(),
    params,
    cv=5   # 5-fold cross validation
)

# Train with different combinations
grid.fit(X_train, y_train)

# Get best model
best_model = grid.best_estimator_

# Predict using best model
best_pred = best_model.predict(X_test)

print("Best Parameters Found:", grid.best_params_)
print(classification_report(y_test, best_pred))

# Predict
pred = model.predict(X_test)

print("Model Performance:\n")
print(classification_report(y_test, pred))