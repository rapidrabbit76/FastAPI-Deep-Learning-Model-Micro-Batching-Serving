import typing as T
from fastapi import (
    Depends,
    File,
    UploadFile,
    HTTPException,
    status,
    Body,
)
from fastapi_restful.cbv import cbv

from fastapi_restful.inferring_router import InferringRouter
from fastapi.logger import logger
from PIL import Image

from ..settings import get_settings
from ..services import Resnet50Service
from ..dependencies import load_classes
from ..schema import PredictTag


router = InferringRouter()
setting = get_settings()


@cbv(router)
class Resnet:
    svc: Resnet50Service = Depends()

    @router.post("/predict/embedding", response_model=T.List[float])
    def predict_embedding(
        self,
        image: UploadFile = File(...),
    ):
        logger.info("------------- Embedding Start -----------")
        image = self.imread(image)
        embedding = self.svc.predict_embedding(image)
        logger.info("------------- Embedding Done -----------")
        return embedding

    @router.post("/predict/tag", response_model=T.List[PredictTag])
    def predict_tag(
        self,
        image: UploadFile = File(...),
        k: int = Body(5, embed=True),
    ):
        logger.info("------------- Tagger Start -----------")
        image = self.imread(image)
        tags = self.svc.predict_tags(image, k)
        logger.info("------------- Tagger Done -----------")
        return tags

    @router.get("/tags", response_model=T.List[str])
    def get_tags(self) -> T.List[str]:
        return load_classes()

    @staticmethod
    def imread(image):
        try:
            image = Image.open(image.file).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail=f"""{image.filename} is not image file, {e} """,
            )
        return image
