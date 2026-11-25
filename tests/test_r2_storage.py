import io
import json

import pytest
from botocore.exceptions import ClientError

import r2_storage


def _error(code, status):
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "op")


class FakeClient:
    def __init__(self, data=None, conflicts=0):
        self.data = data
        self.conflicts = conflicts
        self.writes = []  # data.json put_object calls
        self.backups = 0

    def get_object(self, Bucket, Key):
        if self.data is None:
            raise _error("NoSuchKey", 404)
        return {"Body": io.BytesIO(json.dumps(self.data).encode()), "ETag": '"abc"'}

    def put_object(self, **kwargs):
        if kwargs["Key"] != "data.json":
            self.backups += 1
            return
        self.writes.append(kwargs)
        if self.conflicts:
            self.conflicts -= 1
            raise _error("PreconditionFailed", 412)
        self.data = json.loads(kwargs["Body"])


@pytest.fixture
def use(monkeypatch):
    def _use(client):
        monkeypatch.setattr(r2_storage, "_client", lambda: client)
        monkeypatch.setattr(r2_storage, "list_files", lambda prefix: [])
        return client
    return _use


def _add(data):
    data["books"].append("x")


def test_first_write_requires_missing_object(use):
    client = use(FakeClient())
    r2_storage.update_data(_add)
    assert client.writes[0]["IfNoneMatch"] == "*"
    assert client.data["books"] == ["x"] and client.backups == 0


def test_existing_data_is_backed_up_and_matched(use):
    client = use(FakeClient({"books": []}))
    r2_storage.update_data(_add)
    assert client.writes[0]["IfMatch"] == '"abc"'
    assert client.backups == 1


def test_conflict_is_retried_on_fresh_data(use):
    client = use(FakeClient({"books": []}, conflicts=2))
    r2_storage.update_data(_add)
    assert len(client.writes) == 3
    assert client.data["books"] == ["x"]  # applied once, not three times


def test_conflict_gives_up(use):
    client = use(FakeClient({"books": []}, conflicts=3))
    with pytest.raises(ClientError):
        r2_storage.update_data(_add)
    assert client.data == {"books": []}
