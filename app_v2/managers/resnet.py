import typing as T
from functools import lru_cache

import torch
from service_streamer import ManagedModel, Streamer
from fastapi.logger import logger
from ..settings import get_settings

env = get_settings()


class ResnetTaggerModelManager(ManagedModel):
    def init_model(self):
        classifier = torch.jit.load(
            env.DL_CLASSIFIER_MODEL_PATH, map_location="cpu"
        )
        self.classifier = classifier.eval().to(env.CUDA_DEVICE)

    @torch.inference_mode()
    def predict(self, inputs: T.List[torch.Tensor]) -> T.List[torch.Tensor]:
        logger.info(f"batch size: {len(inputs)}")
        results = []
        try:
            batch = torch.cat(inputs, 0).to(env.CUDA_DEVICE)
            print("batch_size:", batch.shape)
            pred = self.classifier(batch)
            prob = torch.softmax(pred, dim=1)
            prob = prob.cpu()
            results = [output for output in prob]
        except Exception as e:
            logger.error(f"Error {self.__class__.__name__}: {e}")
        return results


class ResnetEmbeddingModelManager(ManagedModel):
    def init_model(self):
        embedding_model = torch.jit.load(
            env.DL_EMBEDDING_MODEL_PATH, map_location="cpu"
        )
        self.embedding_model = embedding_model.eval().to(env.CUDA_DEVICE)

    @torch.inference_mode()
    def predict(self, inputs: T.List[torch.Tensor]) -> T.List[torch.Tensor]:
        logger.info(f"batch size: {len(inputs)}")
        results = []
        try:
            batch = torch.cat(inputs, 0).to(env.CUDA_DEVICE)
            print("batch_size:", batch.shape)
            outputs = self.embedding_model(batch)
            outputs = outputs.cpu()
            results = [output for output in outputs]
        except Exception as e:
            logger.error(f"Error {self.__class__.__name__}: {e}")
        return results


@lru_cache(maxsize=1)
def get_resnet_embedding_streamer():
    streamer = Streamer(
        ResnetEmbeddingModelManager,
        batch_size=env.MB_BATCH_SIZE,
        max_latency=env.MB_MAX_LATENCY,
        worker_num=env.MB_WORKER_NUM,
        cuda_devices=env.CUDA_DEVICES,
    )
    return streamer


@lru_cache(maxsize=1)
def get_resnet_classifier_streamer():
    streamer = Streamer(
        ResnetTaggerModelManager,
        batch_size=env.MB_BATCH_SIZE,
        max_latency=env.MB_MAX_LATENCY,
        worker_num=env.MB_WORKER_NUM,
        cuda_devices=env.CUDA_DEVICES,
    )
    return streamer
