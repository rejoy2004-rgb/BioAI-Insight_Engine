import shap
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess

X, y, scaler = preprocess()

model = RandomForestClassifier()
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)