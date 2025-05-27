#!/usr/bin/env python3

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ─── Configuration ────────────────────────────────────────────────

# TODO: change this to your SageMaker execution role ARN
role = "arn:aws:iam::058264388491:role/service-role/AmazonSageMaker-ExecutionRole-20250523T190106"

# S3 location of the freshly‐built model.tar.gz
model_data_s3_uri = "s3://retail-recommender/model.tar.gz"

# The SageMaker endpoint name you want to create (must be unique per region)
endpoint_name = "retail-recommender-endpoint"

# AWS region where your S3 bucket and SageMaker live
region = "eu-central-1"

# Instance type & count for hosting
instance_type = "ml.m5.large"
initial_instance_count = 1

# ─── Session & Clients ───────────────────────────────────────────

boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# ─── Model Definition ────────────────────────────────────────────

# We're using the prebuilt SKLearn container (v0.24-1 corresponds to sklearn 0.24.x)
model = SKLearnModel(
    model_data=model_data_s3_uri,
    role=role,
    entry_point="inference.py",
    source_dir="code",
    framework_version="1.0-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
)

# ─── Deploy ───────────────────────────────────────────────────────

print(f"Deploying endpoint `{endpoint_name}` with model {model_data_s3_uri} …")
predictor = model.deploy(
    initial_instance_count=initial_instance_count,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
)

print("✅ Deployment initiated.")
print("Endpoint name:", predictor.endpoint_name)