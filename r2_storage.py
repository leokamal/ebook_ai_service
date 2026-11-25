"""Thin wrapper around Cloudflare R2 (S3 API), including the data.json "database".

Required environment variables:
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY  (server-side only)
Optional:
  R2_BUCKET   bucket name (default: "ebook-ai")
"""
import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache

logger = logging.getLogger("uvicorn.error")

DATA_KEY = "data.json"
BACKUP_PREFIX = "backups"
KEEP_BACKUPS = 10
WRITE_ATTEMPTS = 3
DELETE_BATCH = 1000  # S3 delete_objects limit


def bucket_name() -> str:
    return os.environ.get("R2_BUCKET") or "ebook-ai"


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    account_id = os.environ.get("R2_ACCOUNT_ID")
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (account_id and key_id and secret):
        raise RuntimeError("R2_ACCOUNT_ID, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
            # newer botocore adds checksums by default, which S3-compatible services may reject
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _error_status(error):
    return getattr(error, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")


# ------------------------------------------------------------------------------- objects
def list_objects(prefix: str) -> list[dict]:
    """Every object under the folder `prefix` as {"key", "size", "etag"}."""
    prefix = prefix.strip("/")
    pages = _client().get_paginator("list_objects_v2").paginate(
        Bucket=bucket_name(), Prefix=f"{prefix}/" if prefix else "")
    return [
        {"key": o["Key"], "size": o["Size"], "etag": o["ETag"].strip('"')}
        for page in pages
        for o in page.get("Contents", [])
        if not o["Key"].endswith("/")  # folder placeholders
    ]


def list_files(prefix: str) -> list[str]:
    return [o["key"] for o in list_objects(prefix)]


def download_bytes(key: str) -> bytes:
    return _client().get_object(Bucket=bucket_name(), Key=key)["Body"].read()


def download_file(key: str, local_path: str) -> None:
    """Stream an object to disk."""
    _client().download_file(bucket_name(), key, local_path)


def upload_file(key: str, local_path: str) -> None:
    _client().upload_file(local_path, bucket_name(), key)


def delete_files(keys: list[str]) -> None:
    for i in range(0, len(keys), DELETE_BATCH):
        response = _client().delete_objects(
            Bucket=bucket_name(),
            Delete={"Objects": [{"Key": k} for k in keys[i:i + DELETE_BATCH]], "Quiet": True},
        )
        errors = response.get("Errors")
        if errors:
            raise RuntimeError(f"Failed to delete {len(errors)} object(s), first: {errors[0].get('Key')}")


# ----------------------------------------------------------------------------- data.json
def read_data():
    """Return (data, etag), or (None, None) if data.json does not exist yet."""
    from botocore.exceptions import ClientError

    try:
        response = _client().get_object(Bucket=bucket_name(), Key=DATA_KEY)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None, None
        raise
    return json.loads(response["Body"].read()), response["ETag"]


def _backup(data):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    _client().put_object(
        Bucket=bucket_name(),
        Key=f"{BACKUP_PREFIX}/data-{stamp}.json",
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode(),
        ContentType="application/json",
    )
    try:
        delete_files(sorted(list_files(BACKUP_PREFIX))[:-KEEP_BACKUPS])
    except Exception as e:
        logger.warning("Could not prune old backups: %s", e)


def update_data(mutate) -> dict:
    """Read data.json, apply `mutate(data)` in place, write it back only if unchanged meanwhile.

    `mutate` may run more than once, each time on freshly read data.
    """
    from botocore.exceptions import ClientError

    for attempt in range(1, WRITE_ATTEMPTS + 1):
        data, etag = read_data()
        if data is None:
            data = {"schema_version": 1, "books": [], "index": None}
            condition = {"IfNoneMatch": "*"}
        else:
            _backup(data)
            condition = {"IfMatch": etag}
        mutate(data)
        data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            _client().put_object(
                Bucket=bucket_name(),
                Key=DATA_KEY,
                Body=json.dumps(data, ensure_ascii=False, indent=2).encode(),
                ContentType="application/json",
                **condition,
            )
            return data
        except ClientError as e:
            if _error_status(e) not in (409, 412) or attempt == WRITE_ATTEMPTS:
                raise
            logger.warning("data.json changed while writing; retrying (%d/%d)", attempt, WRITE_ATTEMPTS)
