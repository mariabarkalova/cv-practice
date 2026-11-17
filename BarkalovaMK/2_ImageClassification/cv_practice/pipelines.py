#режимы запуска
import csv
import logging
from argparse import Namespace

import numpy as np
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from .bow import BagOfWords
from .data import LandmarksDataLoader
from .features import FeatureExtractor
from .images import read_image

LOGGER = logging.getLogger(__name__)
CPU_THREADS = 4
LOG_INTERVAL = 100


#построение словаря K-средних, гистограмм, обучение классификатора и сохранение модели
def run_train(args: Namespace) -> None:
    samples = LandmarksDataLoader(args.data).load_split("trainval")
    model_path = args.model.resolve()
    if any(model_path == sample.path.resolve() for sample in samples):
        raise ValueError("Путь --model совпадает с обучающим изображением.")

    labels = [sample.label for sample in samples]
    class_count = len(set(labels))
    if class_count < 2:
        raise ValueError("Для обучения нужны изображения хотя бы двух классов.")

    LOGGER.info(
        "Обучающая выборка содержит %d изображений и %d классов",
        len(samples),
        class_count,
    )
    LOGGER.info("Извлечение дескрипторов методом %s", args.extractor)
    extractor = FeatureExtractor.create(args.extractor, device=args.device)
    model = BagOfWords(extractor)
    descriptor_sets = []

    with threadpool_limits(limits=CPU_THREADS):
        for index, sample in enumerate(samples, start=1):
            descriptors = extractor.detect_and_compute(read_image(sample.path))
            descriptor_sets.append(descriptors) #склад в список матрицы дексрипт
            if index % LOG_INTERVAL == 0 or index == len(samples):
                LOGGER.info(
                    "Прочитано %d/%d обучающих изображений", index, len(samples)
                )

        empty_count = sum(len(descriptors) == 0 for descriptors in descriptor_sets)
        if empty_count:
            LOGGER.warning("Пустая гистограмма у %d обучающих изображений", empty_count)
        #обучение модели
        model.fit(descriptor_sets,
            labels,
            words=args.words,
            classifier=args.classifier,
            max_descriptors=args.max_descriptors,
            seed=args.seed,
        )

    model.save_model(args.model)
    LOGGER.info("Модель сохранена в %s", model_path)

#тестирование
#загрузка модели, оценка точности на тестовой выборке
def run_test(args: Namespace) -> None:
    samples = LandmarksDataLoader(args.data).load_split("test") #Загружает тестовые картинки
    labels = [sample.label for sample in samples]
    if args.predictions:
        predictions_path = args.predictions.resolve()
        if predictions_path == args.model.resolve():
            raise ValueError("Укажите разные файлы в --predictions и --model.")
        if any(predictions_path == sample.path.resolve() for sample in samples):
            raise ValueError("Путь --predictions совпадает с тестовым изображением.")

    model = BagOfWords.load_model(args.model, device=args.device) #Загр обученную модель из файла
    unknown = sorted(set(labels).difference(model.classes))
    if unknown:
        raise ValueError(f"В модели отсутствуют тестовые классы: {', '.join(unknown)}.")

    LOGGER.info("Тестирование модели с извлекателем %s", model.extractor.name)
    predictions = []
    empty_count = 0
    with threadpool_limits(limits=CPU_THREADS):
        for index, sample in enumerate(samples, start=1):
            label, histogram = model.predict(read_image(sample.path))
            predictions.append(label) #Полученные предсказания складываются в список
            empty_count += not np.any(histogram)
            if index % LOG_INTERVAL == 0 or index == len(samples):
                LOGGER.info(
                    "Классифицировано %d/%d тестовых изображений", index, len(samples)
                )

    if empty_count:
        LOGGER.warning("Пустая гистограмма у %d тестовых изображений", empty_count)
    correct = int(accuracy_score(labels, predictions, normalize=False))
    LOGGER.info(
        "Точность на test %.2f%% (%d из %d)",
        100 * correct / len(samples),
        correct,
        len(samples),
    )

    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("image", "label", "prediction"))
            writer.writerows(
                (
                    sample.path.relative_to(args.data).as_posix(),
                    sample.label,
                    prediction,
                )
                for sample, prediction in zip(samples, predictions)
            )
        LOGGER.info("Предсказания сохранены в %s", args.predictions.resolve())


#Предсказание одной картинки с визуализацией гистограммы
def run_predict(args: Namespace) -> None:
    from matplotlib import pyplot as plt

    from .visualization import plot_histogram, visualize_image_with_histogram

    paths = (args.model, args.image, args.output, args.histogram_output)
    resolved = [path.resolve() for path in paths if path is not None]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Изображение, модель и графики должны иметь разные пути.")

    image = read_image(args.image)
    model = BagOfWords.load_model(args.model, device=args.device)
    with threadpool_limits(limits=CPU_THREADS):
        label, histogram = model.predict(image)

    LOGGER.info("Предсказанный класс — %s", label)
    if not np.any(histogram):
        LOGGER.warning("Извлекатель вернул пустую гистограмму.")

    figures = [
        visualize_image_with_histogram(image, histogram, label, save_to=args.output)
    ]
    LOGGER.info("Изображение с гистограммой сохранено в %s", args.output.resolve())
    if args.histogram_output:
        figures.append(plot_histogram(histogram, save_to=args.histogram_output))
        LOGGER.info("Гистограмма сохранена в %s", args.histogram_output.resolve())

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
