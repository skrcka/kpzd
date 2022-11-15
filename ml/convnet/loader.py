import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tqdm
from PIL import Image
import tensorflow as tf


class CatsVsDogsLoader:
    def __init__(self, paths: List[Path], image_size: Tuple[int, int]):
        images = []
        labels = []
        for path in tqdm.tqdm(paths):
            image = Image.open(path)
            image = image.resize(image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            images.append(image)

            if path.stem.startswith("cat"):
                label = [1.0, 0.0]
            else:
                label = [0.0, 1.0]
            labels.append(label)

        self.labels = np.array(labels, dtype=np.float32)
        self.images = np.array(images)


def load_example(path: Path, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    image = image.resize(image_size)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0

    if path.stem.startswith("cat"):
        label = [1.0, 0.0]
    else:
        label = [0.0, 1.0]
    label = np.array(label, dtype=np.float32)
    return (image, label)


class LazyCatsVsDogsLoader(tf.keras.utils.Sequence):
    def __init__(self, paths: List[Path], image_size: Tuple[int, int], batch_size: int):
        self.paths = paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.indices = list(range(len(self.paths)))
        self.random = random.Random()
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, index: int):
        start = index * self.batch_size
        end = start + self.batch_size

        images = []
        labels = []
        for index in self.indices[start:end]:
            path = self.paths[index]
            (image, label) = load_example(path, image_size=self.image_size)
            images.append(image)
            labels.append(label)
        return (np.array(images), np.array(labels))

    def on_epoch_end(self):
        self.random.shuffle(self.indices)
