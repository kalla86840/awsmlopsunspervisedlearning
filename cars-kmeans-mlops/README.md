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
- `buildspecs/deploy.yml`: CI/CD buildspec for test, train, register, and deploy.
- `infrastructure/cicd.yaml`: CodePipeline and CodeBuild CloudFormation.

## Deploy CI/CD

Deploy the CI/CD stack with your GitHub CodeConnections ARN and repository:

```powershell
aws cloudformation deploy `
  --template-file cars-kmeans-mlops/infrastructure/cicd.yaml `
  --stack-name cars-kmeans-cicd `
  --capabilities CAPABILITY_NAMED_IAM `
  --parameter-overrides `
    ArtifactBucketName=mlopswithsagemaker111 `
    CodeStarConnectionArn=<your-connection-arn> `
    RepositoryId=<owner/repo> `
    BranchName=main `
    SageMakerExecutionRoleArn=<your-sagemaker-execution-role-arn>
```

The pipeline runs `cars-kmeans-mlops/buildspecs/deploy.yml`, which trains, evaluates, registers, and deploys the real-time endpoint.

## Manual Run

```powershell
pip install -r cars-kmeans-mlops/requirements.txt
python cars-kmeans-mlops/pipelines/run_pipeline.py --config cars-kmeans-mlops/config/pipeline.yaml --wait
python cars-kmeans-mlops/scripts/deploy_endpoint.py --config cars-kmeans-mlops/config/pipeline.yaml --wait
python cars-kmeans-mlops/scripts/invoke_endpoint.py --config cars-kmeans-mlops/config/pipeline.yaml
```
