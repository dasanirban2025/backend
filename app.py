from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from model import train_model
import os
app = Flask(__name__)
CORS(app)

model, scaler, features = train_model()

cols = ["engine_id","cycle"] + [f"op_{i}" for i in range(1,4)] + [f"s_{i}" for i in range(1,22)]
test = pd.read_csv("test_FD001.txt", sep=" ", header=None).iloc[:, :len(cols)]
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





# from flask import Flask, jsonify
# from flask_cors import CORS
# import pandas as pd
# from model import train_model

# app = Flask(__name__)
# CORS(app)

# model, scaler, features = train_model()

# cols = ["engine_id","cycle"] + [f"op_{i}" for i in range(1,4)] + [f"s_{i}" for i in range(1,22)]
# test = pd.read_csv(
#     "test_FD001.txt",
#     sep=r"\s+",
#     header=None
# ).iloc[:, :len(cols)]

# test.columns = cols

# @app.route("/engine/<int:eid>")
# def engine_data(eid):
#     df = test[test.engine_id == eid].copy()
#     if df.empty:
#         return jsonify({"error": "Engine ID not found"}), 404

#     X = scaler.transform(df[features])
#     df["RUL"] = model.predict(X).ravel()

#     max_rul = df["RUL"].max()
#     df["health"] = (df["RUL"] / max_rul * 100) if max_rul > 0 else 0

#     return jsonify(df[["cycle","health","RUL"]].to_dict(orient="records"))

# @app.route("/engines")
# def engines():
#     return jsonify(sorted(test.engine_id.unique().tolist()))







# from flask import Flask, jsonify
# from flask_cors import CORS
# import pandas as pd
# from model import train_model
# import os

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Train/load model, scaler, and features
# model, scaler, features = train_model()

# # Define column names
# cols = ["engine_id", "cycle"] + [f"op_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]

# # Load test dataset
# test = pd.read_csv(
#     "test_FD001.txt",
#     sep=r"\s+",
#     header=None
# ).iloc[:, :len(cols)]
# test.columns = cols

# # Endpoint: engine data with RUL and health
# @app.route("/engine/<int:eid>")
# def engine_data(eid):
#     df = test[test.engine_id == eid].copy()
#     if df.empty:
#         return jsonify({"error": "Engine ID not found"}), 404

#     X = scaler.transform(df[features])
#     df["RUL"] = model.predict(X).ravel()

#     max_rul = df["RUL"].max()
#     df["health"] = (df["RUL"] / max_rul * 100) if max_rul > 0 else 0

#     return jsonify(df[["cycle", "health", "RUL"]].to_dict(orient="records"))


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port, debug=True)
