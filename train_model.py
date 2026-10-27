import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# 1. Load and clean dataset
df = pd.read_csv("data/heart.csv")
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(df.mean(), inplace=True)

# ✅ Convert target to binary (0 = no disease, 1 = disease present)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# 2. Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# ✅ XGBoost Model
# -----------------------------
xgb_model = xgb.XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# Save XGBoost model
joblib.dump(xgb_model, "models/xgb_model.pkl")

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

print("\n🔹 XGBoost Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1-Score:", f1_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# -----------------------------
# ✅ Bayesian Neural Network
# -----------------------------
class BayesianNN(nn.Module):
    def __init__(self, input_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, 0.2, train=True)  # Bayesian approximation
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Initialize model
input_dim = X_train.shape[1]
bayesian_model = BayesianNN(input_dim)
optimizer = optim.Adam(bayesian_model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Convert training data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Train Bayesian NN
for epoch in range(100):  # 100 epochs
    optimizer.zero_grad()
    outputs = bayesian_model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Save Bayesian NN model
torch.save(bayesian_model.state_dict(), "models/bayesian_model.pth")

# Evaluate Bayesian NN
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    outputs = bayesian_model(X_test_tensor)
    y_pred_nn = (outputs.numpy() > 0.5).astype(int).flatten()

print("\n🔹 Bayesian NN Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Precision:", precision_score(y_test, y_pred_nn))
print("Recall:", recall_score(y_test, y_pred_nn))
print("F1-Score:", f1_score(y_test, y_pred_nn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn))

print("\n✅ Training complete. Models saved in 'models/' folder.")
# After saving models, add these lines:

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

print("\n🔹 XGBoost Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1-Score:", f1_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Evaluate Bayesian NN
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    outputs = bayesian_model(X_test_tensor)
    y_pred_nn = (outputs.numpy() > 0.5).astype(int).flatten()

print("\n🔹 Bayesian NN Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Precision:", precision_score(y_test, y_pred_nn))
print("Recall:", recall_score(y_test, y_pred_nn))
print("F1-Score:", f1_score(y_test, y_pred_nn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn))

