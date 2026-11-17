#нейросети
from abc import abstractmethod
from pathlib import Path

import cv2
import torch
from PIL import Image as PilImage
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ( #библиотека, где лежат готовые архит-ры нейросетей
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    WeightsEnum,
    mobilenet_v3_large,
    resnet18,
)

from cv_practice.images import Image

from .base import DescriptorMatrix, FeatureExtractor
#папка, куда скачиваются все веса
CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints"


def _pretrained_state(weights: WeightsEnum) -> dict[str, Tensor]:
    try:
        return weights.get_state_dict(
            model_dir=str(CHECKPOINTS_DIR),
            map_location="cpu",
            check_hash=True,
            weights_only=True,
        )
    except OSError as error:
        raise RuntimeError(
            f"Не удалось получить веса. Загрузите {weights.url} "
            f"в папку {CHECKPOINTS_DIR}. Причина — {error}"
        ) from error


#класс содержит то, что одинаково для всех нейросетей
class CnnExtractorBase(FeatureExtractor):
    #Инициализация (Создание сети)
    def __init__(self, *, device: str = "cpu") -> None:
        if device == "auto": #проверяется на чем считать     
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA недоступна, выберите --device cpu.")
        self.device = torch.device(device)
        backbone, self.preprocess = self._load_backbone()
        self.backbone = backbone.to(self.device).eval().requires_grad_(False)

    @abstractmethod
    def _load_backbone(self) -> tuple[nn.Module, nn.Module]:
        """Загружает обученный backbone и преобразование его входа."""

    @torch.inference_mode() #режим «только предсказание». Он отключает автоматич дифференцирование. 
    def detect_and_compute(self, image: Image) -> DescriptorMatrix:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #картинка в RGB, масшт-ся и нормир-ся
        tensor: Tensor = self.preprocess(PilImage.fromarray(rgb))
        activation = self.backbone(tensor.unsqueeze(0).to(self.device))[0]
        # Каждая позиция карты признаков даёт один локальный дескриптор.
        descriptors = activation.permute(1, 2, 0).reshape(-1, self.descriptor_size)
        return F.normalize(descriptors, p=2, dim=1).cpu().numpy()


class ResNet18Extractor(CnnExtractorBase):
    name = "resnet18"
    descriptor_size = 512

    def _load_backbone(self) -> tuple[nn.Module, nn.Module]:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=None)
        model.load_state_dict(_pretrained_state(weights))
        backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        return backbone, weights.transforms()


class MobileNetV3LargeExtractor(CnnExtractorBase):
    name = "mobilenet_v3_large"
    descriptor_size = 960

    def _load_backbone(self) -> tuple[nn.Module, nn.Module]:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=None)
        model.load_state_dict(_pretrained_state(weights))
        return model.features, weights.transforms()
