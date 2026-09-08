"""Train, compare, and export network intrusion classifiers."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "network_flows.csv"
OUTPUTS = ROOT / "outputs"


def make_pipeline(model, columns):
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), columns)
    ])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Dataset missing. Run: python src/generate_data.py")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    features = [c for c in data.columns if c not in {"flow_id", "label"}]
    x_train, x_test, y_train, y_test = train_test_split(
        data[features], data["label"], test_size=0.20, random_state=42,
        stratify=data["label"],
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=800, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
    }
    fitted = {}
    results = {}
    for name, model in models.items():
        pipeline = make_pipeline(model, features)
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        if hasattr(pipeline, "predict_proba"):
            malicious_index = list(pipeline.classes_).index("malicious")
            score = pipeline.predict_proba(x_test)[:, malicious_index]
        else:
            score = pipeline.decision_function(x_test)
            if pipeline.classes_[1] != "malicious":
                score = -score
        results[name] = {
            "accuracy": round(accuracy_score(y_test, predicted), 4),
            "precision": round(precision_score(y_test, predicted, pos_label="malicious"), 4),
            "recall": round(recall_score(y_test, predicted, pos_label="malicious"), 4),
            "f1": round(f1_score(y_test, predicted, pos_label="malicious"), 4),
            "roc_auc": round(roc_auc_score((y_test == "malicious").astype(int), score), 4),
        }
        fitted[name] = pipeline
        print(name, results[name])

    best_name = max(results, key=lambda key: results[key]["f1"])
    best = fitted[best_name]
    best_predictions = best.predict(x_test)
    artifact = {
        "dataset_rows": len(data),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "selection_metric": "f1",
        "selected_model": best_name,
        "models": results,
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(artifact, indent=2) + "\n")
    joblib.dump(best, OUTPUTS / "best_model.joblib")

    metrics = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    ax = metrics[["accuracy", "recall", "f1"]].plot(kind="bar", figsize=(10, 6), ylim=(0.70, 1.0), color=["#2563eb", "#f97316", "#16a34a"])
    ax.set_title("Intrusion Classifier Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "model_comparison.png", dpi=180)
    plt.close()

    matrix = confusion_matrix(y_test, best_predictions, labels=["normal", "malicious"])
    sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", xticklabels=["Normal", "Malicious"], yticklabels=["Normal", "Malicious"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=180)
    plt.close()

    forest = fitted["Random Forest"]
    importance = pd.Series(forest.named_steps["model"].feature_importances_, index=features).nlargest(12).sort_values()
    importance.plot(kind="barh", figsize=(9, 6), color="#2563eb")
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=180)
    plt.close()
    print(f"Selected {best_name}; artifacts written to {OUTPUTS}")


if __name__ == "__main__":
    main()
