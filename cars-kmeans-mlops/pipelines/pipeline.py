import argparse
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def create_pipeline(config_path="config/pipeline.yaml"):
    config = load_config(config_path)
    boto_session = boto3.Session(region_name=config["aws_region"])
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=config["default_bucket"],
    )
    framework_version = config["model"]["framework_version"]
    feature_columns = ",".join(config["model"]["feature_columns"])

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(ROOT_DIR / "src"),
        framework_version=framework_version,
        py_version="py3",
        role=config["sagemaker_execution_role_arn"],
        instance_type=config["model"]["training_instance_type"],
        instance_count=1,
        base_job_name=f"{config['pipeline']['base_job_prefix']}-train",
        sagemaker_session=pipeline_session,
        hyperparameters={
            "feature-columns": feature_columns,
            "category-column": config["model"]["category_column"],
            "n-clusters": config["model"]["n_clusters"],
            "random-state": config["model"]["random_state"],
        },
    )

    train_step = TrainingStep(
        name="TrainKMeans",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=f"s3://{config['default_bucket']}/{config['data']['s3_prefix']}/",
                content_type="text/csv",
            )
        },
    )

    sklearn_image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=config["aws_region"],
        version=framework_version,
        py_version="py3",
        instance_type=config["model"]["training_instance_type"],
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluation_processor = ScriptProcessor(
        image_uri=sklearn_image_uri,
        command=["python3"],
        role=config["sagemaker_execution_role_arn"],
        instance_count=1,
        instance_type=config["model"]["training_instance_type"],
        base_job_name=f"{config['pipeline']['base_job_prefix']}-eval",
        sagemaker_session=pipeline_session,
    )
    evaluation_step = ProcessingStep(
        name="EvaluateKMeans",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=f"s3://{config['default_bucket']}/{config['data']['s3_prefix']}/",
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ],
        code=str(ROOT_DIR / "src" / "evaluate.py"),
        job_arguments=[
            "--feature-columns",
            feature_columns,
            "--category-column",
            config["model"]["category_column"],
        ],
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluation_step.properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    model = SKLearnModel(
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=config["sagemaker_execution_role_arn"],
        entry_point="inference.py",
        source_dir=str(ROOT_DIR / "src"),
        framework_version=framework_version,
        py_version="py3",
        sagemaker_session=pipeline_session,
    )
    register_step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[config["model"]["inference_instance_type"]],
        transform_instances=[config["model"]["inference_instance_type"]],
        model_package_group_name=config["model"]["model_package_group_name"],
        approval_status=config["model"]["approval_status"],
        model_metrics=model_metrics,
    )
    register_step = ModelStep(
        name="RegisterKMeans",
        step_args=register_step_args,
        depends_on=[evaluation_step],
    )

    condition_step = ConditionStep(
        name="CheckClusterQuality",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="clustering_metrics.silhouette_score.value",
                ),
                right=float(config["model"].get("min_silhouette_score", -1.0)),
            ),
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    return Pipeline(
        name=config["pipeline"]["name"],
        steps=[train_step, evaluation_step, condition_step],
        sagemaker_session=pipeline_session,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Create, update, and optionally start the SageMaker pipeline.")
    parser.add_argument("--config", default=str(ROOT_DIR / "config" / "pipeline.yaml"))
    parser.add_argument("--start", action="store_true", help="Start a pipeline execution after upsert.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    pipeline = create_pipeline(args.config)
    pipeline.upsert(role_arn=config["sagemaker_execution_role_arn"])
    print(f"Upserted SageMaker pipeline: {pipeline.name}")
    if args.start:
        execution = pipeline.start()
        print(f"Started pipeline execution: {execution.arn}")


if __name__ == "__main__":
    main()
