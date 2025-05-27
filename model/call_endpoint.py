#!/usr/bin/env python3
import boto3
import json

# ─── Configuration ────────────────────────────────────────────────
endpoint_name = "retail-recommender-endpoint"   # your SageMaker endpoint
region        = "eu-central-1"                  # your AWS region

# ─── Build the SageMaker Runtime client ───────────────────────────
runtime = boto3.client("sagemaker-runtime", region_name=region)

# ─── Define your payload ──────────────────────────────────────────
payload = {
    "item_id":   212,
    "top_n":     5,
    "threshold": 0.75
}

# ─── Invoke the endpoint ──────────────────────────────────────────
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)

# ─── Read & print the response ───────────────────────────────────
result = json.loads(response["Body"].read().decode())
print(json.dumps(result, indent=2))
