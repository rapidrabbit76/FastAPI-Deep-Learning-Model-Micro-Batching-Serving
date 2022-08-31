import typing as T
from functools import lru_cache
import torch


class EmbeddingSteamerMock:
    @torch.inference_mode()
    def predict(self, image: T.List[torch.Tensor]):
        return [torch.rand([512])]


class TaggerSteamerMock:
    @torch.inference_mode()
    def predict(self, image: T.List[torch.Tensor]):
        return [torch.rand([1000])]


@lru_cache(maxsize=1)
def get_embedding_streamer_mock():
    return EmbeddingSteamerMock()


@lru_cache(maxsize=1)
def get_tagger_streamer_mock():
    return TaggerSteamerMock()
