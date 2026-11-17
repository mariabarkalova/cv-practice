#рисование графиков
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .bow import Histogram
from .images import Image


def _draw_histogram(axes: Axes, histogram: Histogram) -> None:
    words = np.arange(histogram.size)
    axes.bar(words, histogram, width=0.9, color="#266d90")
    axes.set(xlabel="Номер визуального слова", ylabel="Доля дескрипторов")
    axes.set_xlim(-0.5, histogram.size - 0.5)
    axes.set_ylim(bottom=0)
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)


def _save(figure: Figure, path: Path | None) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format=path.suffix.lstrip(".") or "png", dpi=150)


def plot_histogram(histogram: Histogram, *, save_to: Path | None = None) -> Figure:
    figure, axes = plt.subplots(figsize=(8, 4), layout="constrained")
    _draw_histogram(axes, histogram)
    axes.set_title("Гистограмма визуальных слов")
    _save(figure, save_to)
    return figure


def visualize_image_with_histogram(
    image: Image, histogram: Histogram, label: str, *, save_to: Path | None = None
) -> Figure:
    figure, (image_axes, histogram_axes) = plt.subplots(
        1, 2, figsize=(12, 5), layout="constrained"
    )
    image_axes.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_axes.set_title("Входное изображение")
    image_axes.axis("off")
    _draw_histogram(histogram_axes, histogram)
    histogram_axes.set_title("BoW-гистограмма")
    figure.suptitle(f"Предсказанный класс — {label}")
    _save(figure, save_to)
    return figure
