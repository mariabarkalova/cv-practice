#чтение картинок
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8] #многомерный массив NumPy


def read_image(path: Path | str) -> Image:
    """Декодирует изобр как uint8 BGR, в том числе по пути с кириллицей."""
    path = Path(path)
    #читаю файл в байтовый массив через path.read_bytes() и декодирую его через cv2.imdecode(). решает проблему с русс буквами в путях
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Пустой файл изображения: {path}")

    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) #превращение байтов в картинку
    except cv2.error as error:
        raise ValueError(f"OpenCV не смог прочитать изображение: {path}") from error
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать изображение: {path}")
    return image
