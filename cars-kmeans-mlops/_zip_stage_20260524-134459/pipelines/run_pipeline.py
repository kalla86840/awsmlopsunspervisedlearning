import argparse
import time
from pathlib import Path

import boto3

from pipeline import create_pipeline, load_config


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Upload data and run the cars K-Means SageMaker pipeline.")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "pipeline.yaml"))
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def upload_dataset(config):
    local_path = Path(config["data"]["local_path"])
    if not local_path.is_absolute():
        local_path = ROOT_DIR / local_path
    local_path = local_path.resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset not found: {local_path}")

    key = f"{config['data']['s3_prefix'].rstrip('/')}/{local_path.name}"
    boto3.Session(region_name=config["aws_region"]).client("s3").upload_file(
        str(local_path),
        config["default_bucket"],
        key,
    )
    return f"s3://{config['default_bucket']}/{key}"


def wait_for_execution(execution, poll_seconds):
    while True:
        description = execution.describe()
        status = description["PipelineExecutionStatus"]
        print(f"Pipeline execution status: {status}")
        if status == "Succeeded":
            return
        if status in {"Failed", "Stopped"}:
            raise RuntimeError(f"Pipeline execution ended with status: {status}")
        time.sleep(poll_seconds)


def main():
    args = parse_args()
    config = load_config(args.config)
    uploaded_uri = upload_dataset(config)
    print(f"Uploaded training data to {uploaded_uri}")

    pipeline = create_pipeline(args.config)
    pipeline.upsert(role_arn=config["sagemaker_execution_role_arn"])
    execution = pipeline.start()
    print(f"Started SageMaker pipeline execution: {execution.arn}")

    if args.wait:
        wait_for_execution(execution, args.poll_seconds)


if __name__ == "__main__":
    main()
