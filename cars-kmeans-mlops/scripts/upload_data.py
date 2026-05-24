import argparse
from pathlib import Path

import boto3
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload the cars CSV dataset to S3 for SageMaker.")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "pipeline.yaml"))
    parser.add_argument("--local-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    local_path = Path(args.local_path or config["data"]["local_path"])
    if not local_path.is_absolute():
        local_path = ROOT_DIR / local_path
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset not found: {local_path}")

    key = f"{config['data']['s3_prefix'].rstrip('/')}/{local_path.name}"
    boto3.Session(region_name=config["aws_region"]).client("s3").upload_file(
        str(local_path),
        config["default_bucket"],
        key,
    )
    print(f"Uploaded {local_path} to s3://{config['default_bucket']}/{key}")


if __name__ == "__main__":
    main()
