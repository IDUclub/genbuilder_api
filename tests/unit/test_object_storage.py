import json

import pytest

from app.infrastructure.object_storage import (
    LocalStorage,
    MinioStorage,
    ObjectNotFoundError,
    ObjectStorageError,
    get_object_storage,
)


LAYER = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"zone": "residential", "название": "Жилой дом"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 60.0], [30.1, 60.0], [30.0, 60.0]]],
            },
        }
    ],
}


def test_local_put_json_round_trips_and_keeps_unicode(tmp_path):
    storage = LocalStorage(str(tmp_path))

    key = "abc123/buildings.geojson"
    storage.put_json(LAYER, key)

    assert storage.exists(key)
    raw = b"".join(storage.open_stream(key))
    assert json.loads(raw.decode("utf-8")) == LAYER
    # ensure_ascii=False keeps Cyrillic readable instead of \uXXXX escapes.
    assert "Жилой дом".encode("utf-8") in raw


def test_local_exists_is_false_for_missing_object(tmp_path):
    storage = LocalStorage(str(tmp_path))

    assert not storage.exists("nope/buildings.geojson")


def test_local_open_stream_yields_chunks(tmp_path):
    storage = LocalStorage(str(tmp_path))
    key = "abc123/buildings.geojson"
    storage.put_json(LAYER, key)

    chunks = list(storage.open_stream(key))

    assert chunks
    assert all(isinstance(chunk, bytes) for chunk in chunks)


def test_local_put_json_refuses_key_escaping_the_root(tmp_path):
    storage = LocalStorage(str(tmp_path / "outputs"))

    with pytest.raises(ObjectStorageError):
        storage.put_json(LAYER, "../escaped.geojson")


def test_local_open_stream_refuses_a_key_outside_the_root(tmp_path):
    storage = LocalStorage(str(tmp_path / "outputs"))
    (tmp_path / "secret.geojson").write_bytes(b"{}")

    with pytest.raises(ObjectStorageError):
        list(storage.open_stream("../secret.geojson"))


def test_local_put_bytes_round_trips_binary_payload(tmp_path):
    storage = LocalStorage(str(tmp_path))
    payload = b"glTF-binary-payload"

    storage.put_bytes(payload, "3d/abc.glb", "model/gltf-binary")

    assert storage.get_bytes("3d/abc.glb") == payload


def test_local_get_bytes_raises_not_found_for_missing_object(tmp_path):
    storage = LocalStorage(str(tmp_path))

    with pytest.raises(ObjectNotFoundError):
        storage.get_bytes("missing.glb")


def test_local_delete_removes_object_and_ignores_missing_one(tmp_path):
    storage = LocalStorage(str(tmp_path))
    storage.put_bytes(b"x", "probe", "application/octet-stream")

    storage.delete("probe")
    storage.delete("probe")

    assert not storage.exists("probe")


def test_local_put_bytes_refuses_key_escaping_the_root(tmp_path):
    storage = LocalStorage(str(tmp_path / "outputs"))

    with pytest.raises(ObjectStorageError):
        storage.put_bytes(b"x", "../escaped.glb", "model/gltf-binary")


class _FakeS3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _minio_with_client(monkeypatch, client):
    minio_error = pytest.importorskip("minio.error")
    monkeypatch.setattr(minio_error, "S3Error", _FakeS3Error)
    storage = MinioStorage("10.32.1.42:9000", "key", "secret", "genbuilder")
    storage._client = client
    return storage


class _MissingObjectClient:
    def get_object(self, bucket, key):
        raise _FakeS3Error("NoSuchKey")


class _BrokenClient:
    def get_object(self, bucket, key):
        raise _FakeS3Error("AccessDenied")


def test_minio_get_bytes_maps_missing_key_to_not_found(monkeypatch):
    storage = _minio_with_client(monkeypatch, _MissingObjectClient())

    with pytest.raises(ObjectNotFoundError):
        storage.get_bytes("facade-library/v1/manifest.json")


def test_minio_get_bytes_keeps_other_errors_distinct_from_not_found(monkeypatch):
    storage = _minio_with_client(monkeypatch, _BrokenClient())

    with pytest.raises(ObjectStorageError) as excinfo:
        storage.get_bytes("facade-library/v1/manifest.json")

    assert not isinstance(excinfo.value, ObjectNotFoundError)


def test_local_backend_is_not_remote(tmp_path):
    assert not LocalStorage(str(tmp_path)).is_remote()


def _minio_env(monkeypatch):
    monkeypatch.setenv("FILESERVER_ENDPOINT", "10.32.1.42:9000")
    monkeypatch.setenv("FILESERVER_ACCESS_KEY", "key")
    monkeypatch.setenv("FILESERVER_SECRET_KEY", "secret")
    monkeypatch.setenv("FILESERVER_BUCKET_NAME", "genbuilder")
    monkeypatch.setenv("FILESERVER_SECURE", "false")


def test_get_object_storage_selects_minio_when_fully_configured(monkeypatch):
    pytest.importorskip("minio")
    _minio_env(monkeypatch)
    get_object_storage.cache_clear()

    try:
        storage = get_object_storage()
        assert isinstance(storage, MinioStorage)
        assert storage.is_remote()
    finally:
        get_object_storage.cache_clear()


def test_get_object_storage_falls_back_to_local_without_credentials(
    monkeypatch, tmp_path
):
    for key in (
        "FILESERVER_ENDPOINT",
        "FILESERVER_ACCESS_KEY",
        "FILESERVER_SECRET_KEY",
        "FILESERVER_BUCKET_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    get_object_storage.cache_clear()

    try:
        storage = get_object_storage()
        assert isinstance(storage, LocalStorage)
        assert not storage.is_remote()
    finally:
        get_object_storage.cache_clear()


def test_get_object_storage_refuses_a_half_configured_backend(monkeypatch, tmp_path):
    """Falling back to local disk in production would break links on restart."""
    monkeypatch.setenv("FILESERVER_ENDPOINT", "10.32.1.42:9000")
    monkeypatch.setenv("FILESERVER_BUCKET_NAME", "genbuilder")
    for key in ("FILESERVER_ACCESS_KEY", "FILESERVER_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    get_object_storage.cache_clear()

    try:
        with pytest.raises(ObjectStorageError) as excinfo:
            get_object_storage()
        assert "FILESERVER_ACCESS_KEY" in str(excinfo.value)
        assert "FILESERVER_SECRET_KEY" in str(excinfo.value)
    finally:
        get_object_storage.cache_clear()
