from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from model import train_model

app = Flask(__name__)
CORS(app)

model, scaler, features = train_model()

cols = ["engine_id","cycle"] + [f"op_{i}" for i in range(1,4)] + [f"s_{i}" for i in range(1,22)]
# test = pd.read_csv("test_FD001.txt", sep=" ", header=None).iloc[:, :len(cols)]
test = pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None
).iloc[:, :len(cols)]

test.columns = cols

@app.route("/engine/<int:eid>")
def engine_data(eid):
    df = test[test.engine_id == eid]
    X = scaler.transform(df[features])
    df["RUL"] = model.predict(X)

    max_rul = df["RUL"].max()
    df["health"] = (df["RUL"] / max_rul) * 100

    return jsonify(df[["cycle","health","RUL"]].to_dict(orient="records"))

@app.route("/engines")
def engines():
    return jsonify(sorted(test.engine_id.unique().tolist()))

if __name__ == "__main__":
    app.run(debug=True)
