import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import segmentation_models as sm
import tqdm
from PIL import Image
import tensorflow as tf

# https://www.kaggle.com/datasets/tungxnguyen/corrosionannotated

BACKBONE = "resnet50"
preprocess_input = sm.get_preprocessing(BACKBONE)


class CorrosionLoader:
    def __init__(self, names: List[str], input_directory: Path, label_directory: Path,
                 image_size: Tuple[int, int]):
        images = []
        labels = []

        for name in tqdm.tqdm(names):
            path = input_directory / f"{name}.jpg"
            assert path.is_file()

            # Load input image
            image = Image.open(path)
            image = image.resize(image_size)
            image = np.array(image)
            image = preprocess_input(image)
            images.append(image)

            # Load label
            label_path = label_directory / path.with_suffix(".png").name
            assert label_path.is_file()
            label = Image.open(label_path)
            label = label.resize(image_size)
            label = np.array(label)
            label[label > 0] = 255

            # (256, 256, 1)
            label = label.astype(np.float32).reshape((*image_size, 1))
            label = label / 255.0
            labels.append(label)

        self.images = np.array(images)
        self.labels = np.array(labels)


def show_image(img):
    if img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Test", img)
    cv2.waitKey(0)


def get_train_val_sets(count: int, image_size: Tuple[int, int]):
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    input_directory = Path("dataset/input")
    paths = sorted(Path(path).stem for path in os.listdir(input_directory))
    random.shuffle(paths)
    paths = paths[:count]

    val_ratio = 0.2
    train_paths = paths[:int(len(paths) * (1.0 - val_ratio))]
    val_paths = paths[int(len(paths) * (1.0 - val_ratio)):]

    print(f"Train: {len(train_paths)}")
    print(f"Val: {len(val_paths)}")

    train_ds = CorrosionLoader(train_paths, input_directory, Path("dataset/label"),
                               image_size=image_size)
    val_ds = CorrosionLoader(val_paths, input_directory, Path("dataset/label"),
                             image_size=image_size)
    return (train_ds, val_ds)
