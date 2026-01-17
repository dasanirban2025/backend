import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def train_model():
    cols = ["engine_id","cycle"] + [f"op_{i}" for i in range(1,4)] + [f"s_{i}" for i in range(1,22)]
    # train = pd.read_csv("train_FD001.txt", sep=" ", header=None).iloc[:, :len(cols)]
    train = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
).iloc[:, :len(cols)]

    train.columns = cols

    rul = train.groupby("engine_id")["cycle"].max().reset_index()
    rul.columns = ["engine_id","max"]
    train = train.merge(rul, on="engine_id")
    train["RUL"] = train["max"] - train["cycle"]
    train.drop("max", axis=1, inplace=True)

    features = [c for c in train.columns if c not in ["engine_id","cycle","RUL"]]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(train[features])
    y = train["RUL"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, scaler, features
