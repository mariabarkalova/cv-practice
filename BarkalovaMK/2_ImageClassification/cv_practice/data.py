#загрузчик данных
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
CLASS_DIRECTORY = re.compile(r"[0-9]{2}_.+")
IMAGE_SUFFIXES = {
    ".bmp",
    ".jpe",
    ".jpeg",
    ".jfif",
    ".jpg",
    ".png",
}


@dataclass(frozen=True)
#Это простейший контейнер
class ImageSample:
    path: Path #путь к файлу
    label: str #правильный ответ напр 17_MoscowStBasilsCathedral)


class DataLoader(ABC):
    @abstractmethod
    def load_split(self, split: str) -> list[ImageSample]:
        """Возвращает пути к изображениям и их метки для заданной выборки."""


class LandmarksDataLoader(DataLoader):
    """Читает выборки trainval и test из набора, выданного к работе."""

    SOURCES = ("ExtDataset", "NNSUDataset") #папки-источника
    SPLITS = ("trainval", "test") #две выборки

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def load_split(self, split: str) -> list[ImageSample]:
        if split not in self.SPLITS:
            raise ValueError(f"Неизвестная выборка: {split!r}")

        split_path = self.root / split
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Не найдена папка {split_path}."
            )

        samples: list[ImageSample] = []
        for source in self.SOURCES:
            source_path = split_path / source
            if not source_path.is_dir():
                LOGGER.warning("Пропущен отсутствующий источник %s", source_path)
                continue
            #списковые включения
            source_samples = [
                ImageSample(path, class_path.name)
                for class_path in sorted(source_path.iterdir())
                if class_path.is_dir() and CLASS_DIRECTORY.fullmatch(class_path.name)
                for path in sorted(class_path.rglob("*"))
                if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
            ]
            LOGGER.info(
                "%s/%s содержит %d изображений", split, source, len(source_samples)
            )
            samples.extend(source_samples) #Добавляет их в общий список samples

        if not samples:
            raise ValueError(f"В {split_path} не найдено изображений.")
        return samples
