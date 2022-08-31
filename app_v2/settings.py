import typing as T
from functools import lru_cache
from pydantic import BaseSettings


class ModelSettings(BaseSettings):
    DL_EMBEDDING_MODEL_PATH = "model_store/embedding.zip"
    DL_CLASSIFIER_MODEL_PATH = "model_store/classifier.zip"
    CLASSES_PATH = "imagenet_classes.txt"


class MicroBatchSettings(BaseSettings):
    MB_BATCH_SIZE = 64
    MB_MAX_LATENCY = 0.2  # Sec
    MB_WORKER_NUM = 1


class DeviceSettings(BaseSettings):
    CUDA_DEVICE = "cuda"
    CUDA_DEVICES = [0]


class Settings(
    ModelSettings,
    MicroBatchSettings,
    DeviceSettings,
):
    CORS_ALLOW_ORIGINS: T.List[str] = [
        "*",
    ]


@lru_cache()
def get_settings():
    setting = Settings()
    return setting
