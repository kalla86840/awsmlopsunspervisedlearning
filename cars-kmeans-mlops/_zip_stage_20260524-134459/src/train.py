import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-columns", default="age,miles,debt,income,sales")
    parser.add_argument("--category-column", default="gender")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/raw"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return parser.parse_args()


def resolve_csv(path):
    path = Path(path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return csv_files[0]
    return path


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
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_data_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(resolve_csv(args.train))
    feature_columns = [column.strip() for column in args.feature_columns.split(",") if column.strip()]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from the data: {missing}")

    x = frame[feature_columns]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)),
        ]
    )
    labels = model.fit_predict(x)
    scored_frame = frame.copy()
    scored_frame["cluster"] = labels
    scaled_x = model.named_steps["scaler"].transform(x)

    metrics = {
        "inertia": float(model.named_steps["kmeans"].inertia_),
        "silhouette_score": float(silhouette_score(scaled_x, labels)),
        "n_clusters": int(args.n_clusters),
        "training_rows": int(len(frame)),
        "features": feature_columns,
        "cluster_sizes": scored_frame["cluster"].value_counts().sort_index().astype(int).to_dict(),
        **category_profile(scored_frame, args.category_column),
    }

    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "category_column": args.category_column,
                "n_clusters": args.n_clusters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
