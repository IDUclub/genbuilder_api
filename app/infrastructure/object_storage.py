"""Object storage for generated geo layers, with a MinIO backend and a local
filesystem fallback.

Objects are addressed by key. There is no database mapping a result to a path:
the key is derived from the result id, so any caller that knows the id can
address the object without a lookup.

The backend is chosen from the environment at first use: MinIO when endpoint,
bucket and both keys are set, otherwise the local backend so development and
tests work without any infrastructure.

Objects are never served directly to the browser: MinIO lives on a private
network, so the API streams bytes through :func:`open_stream` instead of handing
out presigned URLs.
"""
from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

_MINIO_SCHEME = "minio://"
_CHUNK_SIZE = 64 * 1024
_CONTENT_TYPE = "application/geo+json"
DEFAULT_REGION = "us-east-1"


class ObjectStorageError(RuntimeError):
    """A stored object could not be written or read."""


class ObjectStorage(ABC):
    """Backend for generated geo-layer blobs."""

    @abstractmethod
    def is_remote(self) -> bool:
        """Whether stored paths are remote (MinIO) or local."""

    @abstractmethod
    def put_json(self, payload: dict[str, Any], object_key: str) -> str:
        """Store ``payload`` as UTF-8 JSON. Returns the canonical stored path."""

    @abstractmethod
    def exists(self, object_key: str) -> bool:
        """Whether the object is still present."""

    @abstractmethod
    def open_stream(self, object_key: str) -> Iterator[bytes]:
        """Yield the object's bytes in chunks, without loading it into memory."""


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class LocalStorage(ObjectStorage):
    """Filesystem-backed storage used when MinIO is not configured."""

    def __init__(self, root: str) -> None:
        self._root = Path(root).resolve()

    def is_remote(self) -> bool:
        return False

    def _resolve(self, object_key: str) -> Path:
        """Resolve a key under the storage root, refusing to escape it."""
        path = (self._root / object_key).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise ObjectStorageError(
                f"Object key resolves outside the storage root: {object_key!r}"
            ) from exc
        return path

    def put_json(self, payload: dict[str, Any], object_key: str) -> str:
        path = self._resolve(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_encode(payload))
        return str(path)

    def exists(self, object_key: str) -> bool:
        return self._resolve(object_key).is_file()

    def open_stream(self, object_key: str) -> Iterator[bytes]:
        path = self._resolve(object_key)
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK_SIZE):
                yield chunk


@contextmanager
def _translated_errors(action: str) -> Iterator[None]:
    """Re-raise backend failures as :class:`ObjectStorageError`.

    The MinIO client raises both ``S3Error`` and raw urllib3 connection errors;
    callers must be able to handle a storage failure without importing either.
    """
    try:
        yield
    except Exception as exc:
        raise ObjectStorageError(f"Object storage failed to {action}: {exc}") from exc


class MinioStorage(ObjectStorage):
    """MinIO / S3-compatible object storage."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        region: str = DEFAULT_REGION,
    ) -> None:
        from minio import Minio

        if not endpoint or not bucket:
            raise ValueError("MinIO endpoint and bucket are required")
        self._bucket = bucket
        # ``region`` must be passed explicitly: without it the client resolves the
        # bucket location first, and that call needs an s3:GetBucketLocation right
        # the scoped service credentials deliberately don't have.
        # The bucket is provisioned out of band; the credentials also carry no
        # CreateBucket right, so nothing is created here.
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )

    def is_remote(self) -> bool:
        return True

    def put_json(self, payload: dict[str, Any], object_key: str) -> str:
        data = _encode(payload)
        with _translated_errors(f"store {object_key}"):
            self._client.put_object(
                self._bucket,
                object_key,
                io.BytesIO(data),
                len(data),
                content_type=_CONTENT_TYPE,
            )
        return f"{_MINIO_SCHEME}{object_key}"

    def exists(self, object_key: str) -> bool:
        from minio.error import S3Error

        try:
            self._client.stat_object(self._bucket, object_key)
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchBucket"):
                return False
            raise ObjectStorageError(f"Object storage failed to stat {object_key}: {exc}") from exc
        except Exception as exc:
            raise ObjectStorageError(f"Object storage failed to stat {object_key}: {exc}") from exc
        return True

    def open_stream(self, object_key: str) -> Iterator[bytes]:
        with _translated_errors(f"read {object_key}"):
            response = self._client.get_object(self._bucket, object_key)
        try:
            yield from response.stream(_CHUNK_SIZE)
        finally:
            response.close()
            response.release_conn()


def _env(key: str) -> str | None:
    """Read an optional env var; ``Config()`` has already loaded the .env file."""
    return os.getenv(key) or None


def _env_flag(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes", "on")


_MINIO_KEYS = (
    "FILESERVER_ENDPOINT",
    "FILESERVER_ACCESS_KEY",
    "FILESERVER_SECRET_KEY",
    "FILESERVER_BUCKET_NAME",
)


@lru_cache(maxsize=1)
def get_object_storage() -> ObjectStorage:
    """Return the configured storage backend (cached for the process lifetime).

    All four MinIO variables select MinIO; none of them select the local
    backend. A partial set is a configuration error and is refused rather than
    silently degraded — falling back to local disk in production would produce
    links that break at the next restart.
    """
    present = {key: _env(key) for key in _MINIO_KEYS}
    set_keys = [key for key, value in present.items() if value]

    if len(set_keys) == len(_MINIO_KEYS):
        logger.info(
            "Object storage: MinIO at {}, bucket {}",
            present["FILESERVER_ENDPOINT"],
            present["FILESERVER_BUCKET_NAME"],
        )
        return MinioStorage(
            endpoint=present["FILESERVER_ENDPOINT"] or "",
            access_key=present["FILESERVER_ACCESS_KEY"] or "",
            secret_key=present["FILESERVER_SECRET_KEY"] or "",
            bucket=present["FILESERVER_BUCKET_NAME"] or "",
            secure=_env_flag("FILESERVER_SECURE"),
            region=_env("FILESERVER_REGION") or DEFAULT_REGION,
        )

    if set_keys:
        missing = [key for key in _MINIO_KEYS if key not in set_keys]
        raise ObjectStorageError(
            "Object storage is half-configured; missing: " + ", ".join(missing)
        )

    root = _env("OUTPUTS_DIR") or "outputs"
    logger.info("Object storage: local filesystem at {}", root)
    return LocalStorage(root)
