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
def load_classes():
    with open(env.CLASSES_PATH, "r") as f:
        CLASSES = [s.strip() for s in f.readlines()]
    return CLASSES
