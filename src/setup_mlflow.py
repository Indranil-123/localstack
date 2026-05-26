import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:5000",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

BUCKET = "mlflow-artifacts"


existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]

if BUCKET not in existing:
    s3.create_bucket(Bucket=BUCKET)
    print(f"Bucket '{BUCKET}' is ready.")
else:
    print(f"Bucket '{BUCKET}' already exists.")
    
    
buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
print(f"Current buckets: {buckets}")
