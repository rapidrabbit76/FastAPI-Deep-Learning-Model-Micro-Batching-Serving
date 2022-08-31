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
    load_classifier_model,
    load_embedding_model,
    load_classes,
)
from ..schema import PredictTag

env = get_settings()


class Resnet50Service:
    def __init__(
        self,
        embedding_model=Depends(load_embedding_model),
        classifier=Depends(load_classifier_model),
        classes=Depends(load_classes),
    ):
        logger.info(f"DI: {self.__class__.__name__}")
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.classes = classes

    @torch.inference_mode()
    def predict_embedding(
        self,
        image: Image.Image,
    ) -> T.List[float]:
        image = self.preprocessing(image)
        image = image.to(env.CUDA_DEVICE)
        output = self.embedding_model(image)
        output = output.cpu().numpy()[0]
        output = output.tolist()
        return output

    @torch.inference_mode()
    def predict_tags(
        self,
        image: Image.Image,
        k: int,
    ) -> T.List[PredictTag]:
        image = self.preprocessing(image)
        image = image.to(env.CUDA_DEVICE)
        embedding = self.embedding_model(image)
        pred = self.classifier(embedding)
        prob = torch.softmax(pred[0], dim=0)
        prob = prob.cpu()
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
