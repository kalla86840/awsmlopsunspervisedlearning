import argparse
import json
from pathlib import Path

import boto3
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Invoke the deployed cars K-Means endpoint.")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "pipeline.yaml"))
    parser.add_argument("--payload", default=str(ROOT_DIR / "samples" / "request.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    payload = Path(args.payload).read_text(encoding="utf-8")
    runtime = boto3.Session(region_name=config["aws_region"]).client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=config["endpoint"]["name"],
        ContentType="application/json",
        Body=payload,
    )
    print(response["Body"].read().decode("utf-8"))


if __name__ == "__main__":
    main()
