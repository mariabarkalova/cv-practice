#классические методы
from abc import abstractmethod

import cv2
import numpy as np
from numpy.typing import NDArray

from cv_practice.images import Image

from .base import DescriptorMatrix, FeatureExtractor

MAX_FEATURES = 500
MAX_IMAGE_SIDE = 960


class ClassicalExtractor(FeatureExtractor):
    def __init__(self) -> None:
        self.detector = self._create_detector()

    @abstractmethod
    def _create_detector(self) -> cv2.Feature2D:
        """Создаёт извлекатель OpenCV."""

    @abstractmethod
    def _encode(self, descriptors: NDArray) -> DescriptorMatrix:
        """Приводит дескрипторы к пространству с евклидовой метрикой."""
    

    def detect_and_compute(self, image: Image) -> DescriptorMatrix:
        #перевод в оттенки серого (ч/б). SIFT и ORB работают с яркостью, цвет не важен.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        longest_side = max(height, width)
        if longest_side > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / longest_side
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

        _, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            return np.empty((0, self.descriptor_size), dtype=np.float32)
        return self._encode(descriptors[:MAX_FEATURES])


class SiftExtractor(ClassicalExtractor):
    name = "sift"
    descriptor_size = 128 #SIFT описывает каждую точку вектором из 128 чисел

    def _create_detector(self) -> cv2.Feature2D:
        return cv2.SIFT_create(nfeatures=MAX_FEATURES)

    def _encode(self, descriptors: NDArray) -> DescriptorMatrix:
        descriptors = descriptors.astype(np.float32, copy=False)
        totals = descriptors.sum(axis=1, keepdims=True)
        return np.sqrt(descriptors / np.maximum(totals, 1e-12))


class OrbExtractor(ClassicalExtractor):
    name = "orb"
    descriptor_size = 256 #ORB описывает точку вектором из 256 чисел (но это не обычные числа, а биты 0 и 1)

    def _create_detector(self) -> cv2.Feature2D:
        return cv2.ORB_create(nfeatures=MAX_FEATURES, WTA_K=2)

    def _encode(self, descriptors: NDArray) -> DescriptorMatrix:
        return np.unpackbits(descriptors, axis=1).astype(np.float32)
