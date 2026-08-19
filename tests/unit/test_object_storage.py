import json

import pytest

from app.infrastructure.object_storage import (
    LocalStorage,
    MinioStorage,
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
