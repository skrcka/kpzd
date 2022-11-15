import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from loader import LazyCatsVsDogsLoader

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def create_model(image_shape: Tuple[int, int]):
    input = tf.keras.Input(shape=(*image_shape, 3))
    """layer = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.keras.activations.relu,
    )(input)
    layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
    layer = tf.keras.layers.Flatten()(layer)"""

    resnet = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights=None,
        pooling="avg",
        input_shape=(*image_shape, 3)
    )(input)
    layer = resnet

    layer = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)(layer)
    output = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(layer)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics="accuracy"
    )
    model.summary()
    return model


image_size = (256, 256)
batch_size = 16

directory = Path("dogs-vs-cats/train")
paths = sorted(directory / p for p in os.listdir(directory))
random.shuffle(paths)
paths = paths[:100000]

val_ratio = 0.2
train_paths = paths[:int(len(paths) * (1.0 - val_ratio))]
val_paths = paths[int(len(paths) * (1.0 - val_ratio)):]

print(f"Train count: {len(train_paths)}")
print(f"Val count: {len(val_paths)}")

train_ds = LazyCatsVsDogsLoader(train_paths, image_size=image_size, batch_size=batch_size)
val_ds = LazyCatsVsDogsLoader(val_paths, image_size=image_size, batch_size=batch_size)

model = create_model(image_size)

model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=10)
