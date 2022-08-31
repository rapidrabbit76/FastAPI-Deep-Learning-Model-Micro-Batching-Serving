import typing as T
import torch
from fastapi import Depends
from fastapi.logger import logger
from PIL import Image
from torchvision.transforms.functional import (
    normalize,
    resize,
    to_tensor,
)
from ..settings import get_settings
from ..dependencies import (
    load_classes,
)
from ..managers import (
    get_resnet_embedding_streamer,
    get_resnet_classifier_streamer,
)
from ..schema import PredictTag

env = get_settings()


class Resnet50Service:
    def __init__(
        self,
        embedding_streamer=Depends(get_resnet_embedding_streamer),
        classifier_streamer=Depends(get_resnet_classifier_streamer),
        classes=Depends(load_classes),
    ):
        logger.info(f"DI: {self.__class__.__name__}")
        self.embedding_streamer = embedding_streamer
        self.classifier_streamer = classifier_streamer
        self.classes = classes

    @torch.inference_mode()
    def predict_embedding(
        self,
        image: Image.Image,
    ) -> T.List[float]:
        image = self.preprocessing(image)
        output = self.embedding_streamer.predict([image])[0]
        output = output.numpy()
        output = output.tolist()
        return output

    @torch.inference_mode()
    def predict_tags(
        self,
        image: Image.Image,
        k: int,
    ) -> T.List[PredictTag]:
        image = self.preprocessing(image)
        embedding = self.embedding_streamer.predict([image])[0]
        embedding = torch.unsqueeze(embedding, dim=0)
        prob = self.classifier_streamer.predict([embedding])[0]
        top_prob, top_catid = torch.topk(prob, k)
        top_prob = [prob.item() for prob in top_prob]
        results = [
            PredictTag(name=self.classes[index], prob=prob)
            for prob, index in zip(top_prob, top_catid)
        ]
        return results

    @staticmethod
    def preprocessing(image: Image.Image) -> torch.Tensor:
        image = resize(image, (244, 244))
        image = to_tensor(image)
        image = normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image.unsqueeze(0)
