"""Score one demonstration record with the exported model."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = joblib.load(ROOT / "outputs" / "best_model.joblib")
    data = pd.read_csv(ROOT / "data" / "network_flows.csv")
    record = data.drop(columns=["flow_id", "label"]).iloc[[0]]
    prediction = model.predict(record)[0]
    print({"flow_id": data.iloc[0]["flow_id"], "prediction": prediction})


if __name__ == "__main__":
    main()
