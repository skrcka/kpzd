import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def load_example(path: Path, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    image = image.resize(image_size)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    # image = tf.keras.applications.resnet_v2.preprocess_input(image)

    if path.stem.startswith("cat"):
        label = 0  # [1.0, 0.0]
    else:
        label = 1  # [0.0, 1.0]
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


def show_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    directory = Path("dogs-vs-cats/train")
    paths = sorted(directory / p for p in os.listdir(directory))
    loader = LazyCatsVsDogsLoader(paths, (512, 512), 2)

    augmentator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant"
    )

    for (x, y) in loader:
        for image in x:
            augmented = augmentator.random_transform(image)
            combined = cv2.hconcat((image, augmented))
            show_image(combined)
