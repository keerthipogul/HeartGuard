import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("data/heart.csv")
print(df.head())

def load_and_preprocess():
    df = pd.read_csv("data/heart.csv")
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("target", axis=1))
    y = df["target"]
    return X, y, scaler