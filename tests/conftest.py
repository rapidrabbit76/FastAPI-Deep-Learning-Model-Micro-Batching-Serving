from pydoc import cli
import pytest

from fastapi.testclient import TestClient
from app_v2.main import create_app
from app_v2.managers import (
    get_resnet_classifier_streamer,
    get_resnet_embedding_streamer,
)

from .mocks import get_embedding_streamer_mock, get_tagger_streamer_mock


@pytest.fixture(scope="session")
def client():
    app = create_app()

    app.dependency_overrides[
        get_resnet_embedding_streamer
    ] = get_embedding_streamer_mock

    app.dependency_overrides[
        get_resnet_classifier_streamer
    ] = get_tagger_streamer_mock

    with TestClient(app=app) as client:
        yield client


@pytest.fixture(scope="function")
def files():
    files = [
        (
            "image",
            (
                "temp.png",
                open("src/01.png", "rb"),
                "image/png",
            ),
        )
    ]
    return files
