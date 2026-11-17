# Классификация изображений с использованием библиотеки OpenCV

## Постановка

Приложение определяет достопримечательность на фотографии по гистограмме 
визуальных слов (Bag of Visual Words, BoW). Оно поддерживает два классических
метода OpenCV, две предобученные модели PyTorch и два классификатора
scikit-learn.

В выданном наборе 1 333 изображения в `trainval` и 2 354 изображения в `test`.
В каждой выборке данные из `ExtDataset` и `NNSUDataset` объединяются по имени
папки достопримечательности. Всего 29 меток, включая `00_Other`, для которой
есть размеченные примеры в обеих выборках. Словарь и классификатор обучаются
на `trainval`, точность вычисляется на `test`.

## Структура проекта

Корневая папка — `2_ImageClassification`. Внутри неё: пакет `cv_practice/` 
с логикой, папка `data/` с датасетом, папка `artifacts/` для сохранения 
моделей и графиков, скрипт `main.py`, файл зависимостей `requirements.txt` 
и описание `README.md`.

2_ImageClassification/
    cv_practice/
        __init__.py
        cli.py              парсер командной строки
        data.py             загрузчик данных (DataLoader)
        images.py           чтение картинок (read_image)
        bow.py              ядро BagOfWords (K-Means, гистограммы)
        pipelines.py        сценарии train, test, predict
        visualization.py    рисование графиков
        features/
            __init__.py
            base.py         интерфейс FeatureExtractor и фабрика
            classical.py    SIFT и ORB
            cnn.py          ResNet18 и MobileNetV3
    data/
        Dataset/
            trainval/
            test/
    main.py                 точка входа
    requirements.txt
    README.md


## Запуск
```
python main.py {train,test,predict} [параметры]
```
| Аргумент           | Описание                                               |
|--------------------|--------------------------------------------------------|
| `train`            | Обучение модели на `trainval`                          |
| `test`             | Оценка качества на `test`                              |
| `predict`          | Предсказание класса для одной картинки                 |
| `--extractor`      | Метод извлечения: `sift`, `orb`, `resnet18`,           |
                     `mobilenet_v3_large` (по умолчанию `mobilenet_v3_large`) |
| `--classifier`     | Классификатор: `svm` или `logreg` (по умолчанию `svm`) |
| `--words`          | Размер словаря K-Means (по умолчанию 256)              |
| `--max-descriptors`| Лимит дескрипторов для K-Means (по умолчанию 24000)    |
| `--model`          | Путь к файлу модели (по умолчанию `artifacts/model.joblib`)|
| `--device`         | Устройство: `cpu`, `cuda` или `auto`                   |
| `--image`          | Путь к картинке (только для `predict`)                 |
| `--output`         | Куда сохранить картинку с гистограммой                 |
| `--show`           | Показать графики на экране                             |





## Команды
(справка)
```
python main.py --help

```
###(Демонстрация режима test)
```
python main.py test
``` 
(eсли хотите сохранить результаты в CSV)
```
python main.py test --predictions artifacts/test_predictions.csv
```


###(Демонстрация режима predict)
```
python main.py predict --image "data/Dataset/test/ExtDataset/17_MoscowStBasilsCathedral/131030121411.jpg" --show
```
```
python main.py predict --image "data/Dataset/test/ExtDataset/22_SirotkinHouse/sirotk.jpg" --show
```


###(Обучение)
```
python main.py train
```
- `train` по умолчанию обучает BoW с MobileNetV3-Large и SVM и сохраняет модель в `artifacts/model.joblib`.

другие варианты
```( Обучение с классическим методом SIFT)
python main.py train --extractor sift --classifier svm --model artifacts/sift_svm.joblib
python main.py test --model artifacts/sift_svm.joblib
```

```(Обучение с классическим методом ORB (и логистической регрессией))
python main.py train --extractor orb --classifier logreg --model artifacts/orb_logreg.joblib
python main.py test --model artifacts/orb_logreg.joblib
```

```(Обучение с нейросетью ResNet18)
python main.py train --extractor resnet18 --classifier svm --model artifacts/resnet18_svm.joblib
python main.py test --model artifacts/resnet18_svm.joblib
```



## Результаты

Все конфигурации обучены на 1 333 снимках и проверены на 2 354 снимках. 
Использованы CPU, Python 3.10.20, словарь из 256 центров, 
лимит 24 000 дескрипторов для K-средних и `--seed 42`. Точность — доля 
снимков, для которых предсказанная метка совпала с меткой папки.


| Извлекатель       | Классификатор | Модель в `artifacts` | Точность на всём `test` |
| ------------------| --------------| -------------------- | ------------------------|
| MobileNetV3-Large | SVM           | `model.joblib`       | 66.91% (1575 из 2354)   |
| ResNet-18         | SVM           | `resnet18_svm.joblib`| 68.44% (1611 из 2354)   |
| SIFT              | SVM           | `sift_svm.joblib`    | 42.14% (992 из 2354)    |
| ORB               | Логистическая | `orb_logreg.joblib`  | 21.75% (512 из 2354)    |
                      регрессия




python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

conda create -n cv_lab2 python=3.12 -y
conda activate cv_lab2
pip install -r requirements.txt










