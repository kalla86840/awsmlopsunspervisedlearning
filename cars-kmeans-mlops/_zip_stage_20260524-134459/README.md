# Cars K-Means MLOps Pipeline On AWS

This package trains and deploys a K-Means clustering model for the cars dataset in `data/raw/cars_1020.csv`. The model creates real-time customer/car-sales segments from numeric behavior fields.

## Dataset

The CSV contains:

- `age`
- `gender`
- `miles`
- `debt`
- `income`
- `sales`

K-Means is unsupervised, so there is no supervised target label. This pipeline uses `gender` as the category column for cluster profiling. It is not used as an input feature; it is used to explain how each cluster is distributed across the selected category.

Model features:

- `age`
- `miles`
- `debt`
- `income`
- `sales`

## What The Pipeline Does

1. Uploads `data/raw/cars_1020.csv` to S3.
2. Runs a SageMaker SKLearn training job using `KMeans`.
3. Standardizes numeric features with `StandardScaler`.
4. Emits clustering metrics:
   - inertia
   - silhouette score
   - cluster sizes
   - `gender` distribution by cluster
5. Runs a SageMaker Processing evaluation step.
6. Registers the model package when `silhouette_score >= min_silhouette_score`.
7. Creates or updates a SageMaker real-time endpoint from the latest approved model package.

## Real-Time Inference Shape

Request:

```json
{
  "instances": [
    {
      "age": 28,
      "miles": 23,
      "debt": 0,
      "income": 4099,
      "sales": 620
    }
  ]
}
```

Response:

```json
{
  "predictions": [
    {
      "cluster": 0,
      "distance_to_centroid": 1.23
    }
  ]
}
```

## Key Files

- `config/pipeline.yaml`: AWS, S3, SageMaker, model, and endpoint settings.
- `src/train.py`: K-Means training and metric generation.
- `src/evaluate.py`: SageMaker Processing evaluation report.
- `src/inference.py`: SageMaker real-time inference functions.
- `pipelines/run_pipeline.py`: uploads data, starts the SageMaker pipeline, and optionally waits.
- `scripts/deploy_endpoint.py`: creates or updates the real-time endpoint.
- `buildspecs/test.yml`: CI/CD unit test buildspec.
- `buildspecs/train-register.yml`: CI/CD SageMaker Pipeline buildspec.
- `buildspecs/deploy-endpoint.yml`: CI/CD endpoint deployment buildspec.
- `infrastructure/codepipeline.yml`: CodePipeline and CodeBuild CloudFormation.

## Deploy CI/CD

Deploy the CI/CD stack with your GitHub CodeConnections ARN and repository:

```bash
export REGION="us-west-1"
export PROJECT="cars-kmeans"
export STACK_NAME="cars-kmeans-cicd"
export ARTIFACT_BUCKET="cars-kmeans-codepipeline-artifacts-659613508664"
export REPO="kalla86840/awsmlopsunsupervisedlearning"
export BRANCH="main"
export CONNECTION_ARN="<your-codeconnections-arn>"
export CODEPIPELINE_ROLE_ARN="arn:aws:iam::659613508664:role/sagemaker-mlops-realtime-codepipeline-role"
export CODEBUILD_ROLE_ARN="arn:aws:iam::659613508664:role/sagemaker-mlops-realtime-codebuild-role"

aws s3 mb "s3://${ARTIFACT_BUCKET}" --region "$REGION"

aws cloudformation deploy \
  --template-file infrastructure/codepipeline.yml \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName="$PROJECT" \
    ArtifactBucketName="$ARTIFACT_BUCKET" \
    CodeStarConnectionArn="$CONNECTION_ARN" \
    FullRepositoryId="$REPO" \
    BranchName="$BRANCH" \
    CodePipelineServiceRoleArn="$CODEPIPELINE_ROLE_ARN" \
    CodeBuildServiceRoleArn="$CODEBUILD_ROLE_ARN"
```

The pipeline runs tests, trains/evaluates/registers the model, then deploys the latest approved package to `cars-kmeans-realtime`.

## Manual Run

```bash
pip install -r requirements.txt
python pipelines/run_pipeline.py --wait
python scripts/deploy_endpoint.py --wait
python scripts/invoke_endpoint.py
```
