import pytest
from fastapi.testclient import TestClient


def test_predict_embedding(client: TestClient, files):
    res = client.post(
        "/predict/embedding",
        files=files,
        headers={},
    )

    assert res.ok

    embedding = res.json()

    assert isinstance(embedding[0], float)
    assert len(embedding) == 512


@pytest.mark.parametrize("k", [i for i in range(1, 100, 2)])
def test_predict_tag(client: TestClient, files, k):
    res = client.post(
        "/predict/tag",
        files=files,
        data={"k": k},
        headers={},
    )

    assert res.ok

    tags = res.json()
    assert isinstance(tags[0], dict)
    assert len(tags) == k


def test_get_tags(client: TestClient):
    res = client.get("/tags")

    assert res.ok

    tags = res.json()
    assert isinstance(tags[0], str)
    assert len(tags) == 1000
