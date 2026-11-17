#парсер командной строки
import argparse
from collections.abc import Sequence
from pathlib import Path

from .bow import CLASSIFIERS
from .features import EXTRACTORS
from .pipelines import run_predict, run_test, run_train

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_DIR / "data" / "Dataset"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts" #папка artifacts, куда по умолчанию сохр модель и графики

#созд-ся парсер
def cli_argument_parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Классификация достопримечательностей по гистограммам визуальных слов."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Подробный журнал и трассировка ошибок."
    )
    #позволяет создавать подкоманды (train, test, predict) без них не запустится
    subparsers = parser.add_subparsers(dest="mode", required=True)


    #общие настр-ки
    #shared — это "шаблон" арг, которые нужны всем трем режимам
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--model",
        type=Path,
        default=ARTIFACTS_DIR / "model.joblib",
        help="Путь к файлу модели.",
    )
    shared.add_argument(
        "--device", #где считать нейросеть (cpu по умолчанию)
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Устройство для CNN. По умолчанию cpu.",
    )


    #Настройки данных
    #шаблон для режимов, которым нужны данные (train и test)
    dataset = argparse.ArgumentParser(add_help=False)
    dataset.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Корень набора с папками trainval и test.",
    )

    
    #Режим train
    train = subparsers.add_parser( #Создается подкоманда train
        "train",
        parents=[shared, dataset], #подтягивает аргументы --model, --device и --data
        help="Обучить словарь и классификатор на trainval.",
    )
    # дальше спец арг для обучения
    train.add_argument(
        "--extractor",
        choices=EXTRACTORS,
        default="mobilenet_v3_large",
        help="Метод извлечения дескрипторов. По умолчанию mobilenet_v3_large.",
    )
    train.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        default="svm",
        help="Модель для BoW-гистограмм. По умолчанию svm.",
    )
    train.add_argument(
        "--words", type=int, default=256, help="Размер словаря. По умолчанию 256."
    )
    train.add_argument(
        "--max-descriptors",
        type=int,
        default=24_000,
        help="Лимит дескрипторов для K-средних. По умолчанию 24000.",
    )
    train.add_argument(
        "--seed", type=int, default=42, help="Случайное зерно. По умолчанию 42."
    )
    train.set_defaults(handler=run_train)


    # Режим test
    test = subparsers.add_parser(
        "test", parents=[shared, dataset], help="Оценить качество на test."
    )
    test.add_argument( #Доб опциональный арг --predictions, чтобы сохр таблицу с рез-ми
        "--predictions", type=Path, help="CSV с результатами для отдельных изображений."
    )
    test.set_defaults(handler=run_test) #Уст-т ф-цию run_test

    
    #Режим predict
    #Подкоманда predict берет только общие настр (shared), без dataset, так как датасет тут не нужен
    predict = subparsers.add_parser(
        "predict",
        parents=[shared], 
        help="Предсказать класс и построить BoW-гистограмму.",
    )
    predict.add_argument(
        "--image", type=Path, required=True, help="Входное изображение."
    )
    predict.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "prediction.png",
        help="Файл с входным изображением и гистограммой.",
    )
    predict.add_argument(
        "--histogram-output", type=Path, help="Файл с отдельной гистограммой."
    )
    predict.add_argument(
        "--show", action="store_true", help="Показать графики средствами Matplotlib."
    )
    predict.set_defaults(handler=run_predict) #Уст-ет ф-цию run_predict

    args = parser.parse_args(argv)
    if args.mode == "train":
        if not 2 <= args.words <= args.max_descriptors:
            train.error("должно выполняться 2 <= --words <= --max-descriptors")
        if not 0 <= args.seed < 2**32:
            train.error("--seed должен быть в диапазоне от 0 до 2**32 - 1")
    return args
