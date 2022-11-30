from typing import Tuple

import tensorflow as tf

from loader import BACKBONE, get_train_val_sets


def create_model(image_size: Tuple[int, int]):
    import segmentation_models as sm

    model = sm.Unet(
        backbone_name=BACKBONE,
        input_shape=(*image_size, 3),
        classes=1,
        encoder_weights=None,
        activation="sigmoid"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        # loss=sm.losses.bce_jaccard_loss,
        loss="binary_crossentropy",
        metrics=[sm.metrics.iou_score, tf.keras.metrics.FalsePositives(),
                 tf.keras.metrics.FalseNegatives()],
    )
    return model


image_size = (256, 256)
(train_ds, val_ds) = get_train_val_sets(10000, image_size=image_size)

batch_size = 8
epochs = 100

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
    tf.keras.callbacks.ModelCheckpoint("models/{val_iou_score:.2f}-{epoch:02d}.hdf5",
                                       monitor="val_iou_score", mode="max")
]

model = create_model(image_size)
model.fit(
    x=train_ds.images,
    y=train_ds.labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_ds.images, val_ds.labels),
    callbacks=callbacks
)
