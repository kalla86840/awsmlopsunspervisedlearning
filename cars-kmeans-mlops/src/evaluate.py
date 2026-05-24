import argparse
import json
import tarfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--test-data", default="/opt/ml/processing/test")
    parser.add_argument("--output-dir", default="/opt/ml/processing/evaluation")
    parser.add_argument("--feature-columns", default="age,miles,debt,income,sales")
    parser.add_argument("--category-column", default="gender")
    return parser.parse_args()


def resolve_csv(path):
    path = Path(path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return csv_files[0]
    return path


def load_model(model_artifact):
    artifact = Path(model_artifact)
    extract_dir = Path("/tmp/cars-kmeans-model")
    extract_dir.mkdir(parents=True, exist_ok=True)
    if artifact.is_dir():
        artifact = artifact / "model.tar.gz"
    with tarfile.open(artifact) as tar:
        tar.extractall(extract_dir)
    return joblib.load(extract_dir / "model.joblib")


def category_profile(frame, category_column):
    if category_column not in frame.columns:
        return {}
    return {
        "category_column": category_column,
        "category_by_cluster": (
            pd.crosstab(frame["cluster"], frame[category_column], normalize="index")
            .round(4)
            .to_dict()
        ),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(resolve_csv(args.test_data))
    feature_columns = [column.strip() for column in args.feature_columns.split(",") if column.strip()]
    model = load_model(args.model_artifact)
    x = frame[feature_columns]
    labels = model.predict(x)
    scored_frame = frame.copy()
    scored_frame["cluster"] = labels
    scaled_x = model.named_steps["scaler"].transform(x)

    report = {
        "clustering_metrics": {
            "inertia": {"value": float(model.named_steps["kmeans"].inertia_)},
            "silhouette_score": {"value": float(silhouette_score(scaled_x, labels))},
            "n_clusters": {"value": int(model.named_steps["kmeans"].n_clusters)},
        },
        "cluster_profile": {
            "cluster_sizes": scored_frame["cluster"].value_counts().sort_index().astype(int).to_dict(),
            **category_profile(scored_frame, args.category_column),
        },
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
