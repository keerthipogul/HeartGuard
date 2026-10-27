import shap

def explain_model(xgb_model, X_sample):
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_sample)
    return shap_values