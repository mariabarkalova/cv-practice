from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from cv_practice.images import Image

DescriptorMatrix = NDArray[np.float32]
CLASSICAL_EXTRACTORS = ("sift", "orb") #классич метод
CNN_EXTRACTORS = ("resnet18", "mobilenet_v3_large") #неуросети
EXTRACTORS = (*CLASSICAL_EXTRACTORS, *CNN_EXTRACTORS)


class FeatureExtractor(ABC):
    name: str
    descriptor_size: int

    @abstractmethod
    def detect_and_compute(self, image: Image) -> DescriptorMatrix:
        """Возвращает матрицу (число дескрипторов, размерность дескриптора)."""

    @staticmethod
    def create(name: str, *, device: str = "cpu") -> FeatureExtractor:
        if name in CLASSICAL_EXTRACTORS:
            from .classical import OrbExtractor, SiftExtractor

            extractors = {"sift": SiftExtractor, "orb": OrbExtractor}
            return extractors[name]()

        if name in CNN_EXTRACTORS:
            from .cnn import MobileNetV3LargeExtractor, ResNet18Extractor

            cnn_extractors = {
                "resnet18": ResNet18Extractor,
                "mobilenet_v3_large": MobileNetV3LargeExtractor,
            }
            return cnn_extractors[name](device=device)

        raise ValueError(f"Неизвестный метод извлечения признаков: {name!r}")
