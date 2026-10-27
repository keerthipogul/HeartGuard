import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np

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

def predict_risk(features, xgb_model, bayesian_model):
    # Convert to numpy
    features = np.array(features).reshape(1, -1)

    # XGBoost prediction
    xgb_pred = xgb_model.predict(features)

    # Bayesian NN prediction
    tensor_features = torch.tensor(features, dtype=torch.float32)
    bayesian_pred = bayesian_model(tensor_features).detach().numpy()

    # Hybrid probability
    final_prob = (xgb_pred + bayesian_pred) / 2
    return final_prob[0]