import argparse
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy the latest approved model package to a SageMaker endpoint.")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "pipeline.yaml"))
    parser.add_argument("--wait", action="store_true", help="Wait until the endpoint is in service.")
    return parser.parse_args()


def latest_approved_package(client, package_group):
    paginator = client.get_paginator("list_model_packages")
    packages = []
    for page in paginator.paginate(
        ModelPackageGroupName=package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    ):
        packages.extend(page["ModelPackageSummaryList"])
    if not packages:
        raise RuntimeError(f"No approved model packages found in group {package_group}.")
    return packages[0]["ModelPackageArn"]


def endpoint_exists(client, endpoint_name):
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def main():
    args = parse_args()
    config = load_config(args.config)
    session = boto3.Session(region_name=config["aws_region"])
    client = session.client("sagemaker")

    endpoint_name = config["endpoint"]["name"]
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_name = f"{endpoint_name}-model-{suffix}"
    endpoint_config_name = f"{endpoint_name}-config-{suffix}"
    model_package_arn = latest_approved_package(
        client,
        config["model"]["model_package_group_name"],
    )

    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=config["sagemaker_execution_role_arn"],
        PrimaryContainer={"ModelPackageName": model_package_arn},
    )
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": int(config["endpoint"]["initial_instance_count"]),
                "InstanceType": config["endpoint"]["instance_type"],
            }
        ],
    )

    if endpoint_exists(client, endpoint_name):
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "Updated"
    else:
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "Created"

    print(f"{action} endpoint {endpoint_name} with model package {model_package_arn}")
    if args.wait:
        client.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
        print(f"Endpoint is in service: {endpoint_name}")


if __name__ == "__main__":
    main()
