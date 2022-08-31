from functools import lru_cache
from fastapi.logger import logger
import torch
from .settings import get_settings


logger.info("---------- dependencies init -------------")
env = get_settings()
torch.set_grad_enabled(False)
logger.info("model loaded Start ")
logger.info("---------- dependencies init done ----------")


@lru_cache(maxsize=1)
def load_embedding_model():
    model = torch.jit.load(
        env.DL_EMBEDDING_MODEL_PATH,
        map_location="cpu",
    ).eval()
    model = model.to(env.CUDA_DEVICE)
    return model


@lru_cache(maxsize=1)
def load_classifier_model():
    model = torch.jit.load(
        env.DL_CLASSIFIER_MODEL_PATH,
        map_location="cpu",
    ).eval()
    model = model.to(env.CUDA_DEVICE)
    return model


@lru_cache(maxsize=1)
def load_classes():
    with open(env.CLASSES_PATH, "r") as f:
        CLASSES = [s.strip() for s in f.readlines()]
    return CLASSES
