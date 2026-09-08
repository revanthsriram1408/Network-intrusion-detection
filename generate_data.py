"""Generate reproducible, synthetic network-flow data for the portfolio demo."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "network_flows.csv"
FEATURES = [
    "duration_ms", "src_bytes", "dst_bytes", "packets_in", "packets_out",
    "avg_packet_size", "syn_count", "ack_count", "rst_count", "fin_count",
    "unique_dst_ports", "failed_connections", "same_service_rate",
    "dst_host_count", "dst_host_srv_count", "error_rate", "srv_error_rate",
    "login_attempts", "privileged_ops", "urgent_packets", "land_flag",
    "wrong_fragment", "hot_indicators", "connection_rate",
]


def build_dataset(rows: int = 120_000, seed: int = 42) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=rows,
        n_features=len(FEATURES),
        n_informative=16,
        n_redundant=4,
        n_clusters_per_class=3,
        weights=[0.72, 0.28],
        class_sep=1.75,
        flip_y=0.022,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-9)
    scales = np.array([
        120_000, 2_000_000, 2_000_000, 500, 500, 1500, 40, 80, 20, 20,
        50, 25, 1, 255, 255, 1, 1, 12, 8, 5, 1, 10, 15, 100,
    ])
    frame = pd.DataFrame(x * scales, columns=FEATURES)
    count_columns = [
        "packets_in", "packets_out", "syn_count", "ack_count", "rst_count",
        "fin_count", "unique_dst_ports", "failed_connections", "dst_host_count",
        "dst_host_srv_count", "login_attempts", "privileged_ops",
        "urgent_packets", "land_flag", "wrong_fragment", "hot_indicators",
    ]
    frame[count_columns] = frame[count_columns].round().astype(int)
    frame.insert(0, "flow_id", [f"FLOW-{i:06d}" for i in range(1, rows + 1)])
    frame["label"] = np.where(y == 1, "malicious", "normal")

    # Add a small, realistic missing-value rate for pipeline validation.
    for column in ("duration_ms", "src_bytes", "dst_bytes", "error_rate"):
        missing = rng.choice(rows, size=max(1, rows // 500), replace=False)
        frame.loc[missing, column] = np.nan
    return frame


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    data.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(data):,} rows to {OUTPUT}")
    print(data["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
