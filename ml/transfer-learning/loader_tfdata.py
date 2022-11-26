import math
import os.path
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from loader import LazyCatsVsDogsLoader


def iterate_examples(paths: List[str], image_size: Tuple[int, int]):
    for path in paths:
        # image = Image.open(path)
        # image = image.resize(image_size)
        # image = np.array(image).astype(np.float32) / 255
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, np.array(image_size))

        yield image


def iterate_labels(paths: List[str]):
    for path in paths:
        if os.path.basename(path).startswith(b"cat"):
            label = [1.0, 0.0]
        else:
            label = [0.0, 1.0]
        label = np.array(label, dtype=np.float32)
        yield label


def create_dataset(paths: List[str], image_size: Tuple[int, int],
                   batch_size: int) -> tf.data.Dataset:
    paths = [str(p) for p in paths]

    image_dataset = tf.data.Dataset.from_generator(iterate_examples, args=[paths, image_size],
                                                   output_types=tf.float32,
                                                   output_shapes=(*image_size, 3))
    label_dataset = tf.data.Dataset.from_generator(iterate_labels, args=[paths],
                                                   output_types=tf.float32,
                                                   output_shapes=(2, ))
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset


if __name__ == "__main__":
    directory = Path("dogs-vs-cats/train")
    paths = sorted(directory / p for p in os.listdir(directory))
    paths = paths[:2000]

    batch_size = 32
    ds = create_dataset(paths, (256, 256), batch_size=batch_size)
    # ds = LazyCatsVsDogsLoader(paths, image_size=(256, 256), batch_size=batch_size)
    ds_iter = iter(ds)

    steps_per_epoch = math.ceil(len(paths) / batch_size)
    print(f"Steps: {steps_per_epoch}")
    for _ in range(5):
        start = time.time()

        ds_iter = iter(ds)
        for _ in range(steps_per_epoch):
            next(ds_iter)

        end = time.time()
        duration = end - start
        print(f"Duration: {duration:.2f} s")
