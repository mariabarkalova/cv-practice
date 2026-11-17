#мешок визуальных слов
import logging
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .features import DescriptorMatrix, FeatureExtractor
from .images import Image

LOGGER = logging.getLogger(__name__)
MODEL_FORMAT = "cv-practice-bow-v1"
CLASSIFIERS = ("svm", "logreg")
Classifier = SVC | LogisticRegression
Histogram = NDArray[np.float32]


class BagOfWords:
    def __init__(self, extractor: FeatureExtractor) -> None:
        self.extractor = extractor
        self.vocabulary: KMeans | None = None
        self.classifier: Classifier | None = None

    def build_vocabulary(self,descriptor_sets: Sequence[DescriptorMatrix],*,
        words: int, max_descriptors: int, seed: int,) -> KMeans:
        """Строит словарь на дескрипторах trainval методом K-средних"""
        if not 2 <= words <= max_descriptors:
            raise ValueError("Должно выполняться 2 <= words <= max_descriptors.")

        offsets = np.cumsum([0, *(len(rows) for rows in descriptor_sets)])
        total = int(offsets[-1])
        if total < words:
            raise ValueError(
                f"Найдено {total} дескрипторов для {words} визуальных слов. "
                "Увеличьте выборку или уменьшите --words."
            )

        if total <= max_descriptors:
            training_rows = np.concatenate(descriptor_sets, axis=0)
        else:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(total, max_descriptors, replace=False))
            bounds = np.searchsorted(selected, offsets)
            training_rows = np.concatenate(
                [
                    rows[selected[left:right] - start]
                    for rows, start, left, right in zip(
                        descriptor_sets, offsets[:-1], bounds[:-1], bounds[1:]
                    )
                    if left < right
                ],
                axis=0,
            )

        LOGGER.info(
            "K-средних обучается на %d из %d дескрипторов", len(training_rows), total
        )
        #Обучение K-Means
        return KMeans(
            n_clusters=words,
            init="k-means++",
            n_init=3,
            max_iter=100,
            random_state=seed,
            algorithm="lloyd",
        ).fit(training_rows)

    @staticmethod
    def _histogram(descriptors: DescriptorMatrix, vocabulary: KMeans) -> Histogram: #дескр-ры одной карт в гист
        if len(descriptors) == 0:
            return np.zeros(vocabulary.n_clusters, dtype=np.float32)

        words = vocabulary.predict(descriptors) #находим к какому слову дескриптор нах-ся ближе
        counts = np.bincount(words, minlength=vocabulary.n_clusters).astype(np.float32) #сколько раз встретилось каждое слово
        return counts / len(descriptors) #нормир

    def compute_histogram(self, descriptors: DescriptorMatrix) -> Histogram:
        vocabulary, _ = self._fitted()
        return self._histogram(descriptors, vocabulary)

    #Обучение всей модели
    def fit(self, descriptor_sets: Sequence[DescriptorMatrix], labels: Sequence[str], *, 
            words: int, classifier: str, max_descriptors: int, seed: int,) -> None:
        """Обучает словарь и классификатор на размеченных наборах дескрипторов."""
        if len(descriptor_sets) != len(labels):
            raise ValueError(
                "Число наборов дескрипторов должно совпадать с числом меток."
            )
        if len(set(labels)) < 2:
            raise ValueError("Для обучения нужны изображения хотя бы двух классов.")

        estimator: Classifier
        if classifier == "svm":
            estimator = SVC(
                C=10.0, kernel="rbf", gamma="scale", class_weight="balanced"
            )
        elif classifier == "logreg":
            estimator = LogisticRegression(
                C=10.0, class_weight="balanced", max_iter=1_000
            )
        else:
            raise ValueError(f"Неизвестный классификатор: {classifier!r}")

        vocabulary = self.build_vocabulary(descriptor_sets, words=words, max_descriptors=max_descriptors, seed=seed)
        histograms = np.stack([self._histogram(rows, vocabulary) for rows in descriptor_sets])
        LOGGER.info("Обучение %s на %d гистограммах", classifier, len(histograms))
        estimator.fit(np.sqrt(histograms), labels)
        self.vocabulary = vocabulary
        self.classifier = estimator

    def _fitted(self) -> tuple[KMeans, Classifier]:
        if self.vocabulary is None or self.classifier is None:
            raise RuntimeError("Сначала обучите модель или загрузите её из файла.")
        return self.vocabulary, self.classifier

    @property
    def classes(self) -> tuple[str, ...]:
        _, classifier = self._fitted()
        return tuple(str(label) for label in classifier.classes_)


    #предсказание
    def predict(self, image: Image) -> tuple[str, Histogram]:
        vocabulary, classifier = self._fitted() #Провер, что модель обучена
        descriptors = self.extractor.detect_and_compute(image) #получаем дескрипторы
        histogram = self._histogram(descriptors, vocabulary)  #дескрипторы в гистогр
        predicted = classifier.predict(np.sqrt(histogram[None, :]))  #Отправляем гист в классификатор -> предсказанный класс
        return str(predicted[0]), histogram

    #сохранение модели
    def save_model(self, path: Path) -> None:
        vocabulary, classifier = self._fitted()
        bundle = {
            "format": MODEL_FORMAT,
            "extractor": self.extractor.name,
            "vocabulary": vocabulary,
            "classifier": classifier,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp_dir:
            temporary = Path(temp_dir) / path.name
            joblib.dump(bundle, temporary, compress=3)
            temporary.replace(path)

    @classmethod
    #загрузка модели
    def load_model(cls, path: Path, *, device: str = "cpu") -> "BagOfWords":
        """Загружает модель, созданную save_model, и её извлекатель признаков"""
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
            raise ValueError(f"Неподдерживаемый формат файла модели: {path}")

        vocabulary = bundle.get("vocabulary")
        classifier = bundle.get("classifier")
        name = bundle.get("extractor")
        if (
            not isinstance(vocabulary, KMeans)
            or not isinstance(classifier, (SVC, LogisticRegression))
            or not isinstance(name, str)
            or getattr(classifier, "n_features_in_", None) != vocabulary.n_clusters
        ):
            raise ValueError(f"Некорректное содержимое файла модели: {path}")

        extractor = FeatureExtractor.create(name, device=device)
        if getattr(vocabulary, "n_features_in_", None) != extractor.descriptor_size:
            raise ValueError(f"Словарь и извлекатель признаков несовместимы: {path}")

        model = cls(extractor)
        model.vocabulary = vocabulary
        model.classifier = classifier
        return model
