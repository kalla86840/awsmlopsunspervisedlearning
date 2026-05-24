import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = ROOT_DIR.parent


def test_train_and_infer_locally(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    data_path = WORKSPACE_DIR / "data" / "raw" / "cars_1020.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "src" / "train.py"),
            "--train",
            str(data_path),
            "--model-dir",
            str(model_dir),
            "--output-data-dir",
            str(output_dir),
            "--feature-columns",
            "age,miles,debt,income,sales",
            "--category-column",
            "gender",
            "--n-clusters",
            "4",
        ],
        check=True,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_clusters"] == 4
    assert "silhouette_score" in metrics
    assert metrics["category_column"] == "gender"

    sys.path.insert(0, str(ROOT_DIR / "src"))
    import inference

    model_bundle = inference.model_fn(str(model_dir))
    request = {
        "instances": [
            {"age": 28, "miles": 23, "debt": 0, "income": 4099, "sales": 620}
        ]
    }
    parsed = inference.input_fn(json.dumps(request), "application/json")
    prediction = inference.predict_fn(parsed, model_bundle)
    assert prediction[0]["cluster"] in {0, 1, 2, 3}
    assert prediction[0]["distance_to_centroid"] >= 0
