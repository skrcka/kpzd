import numpy as np

from loader import get_train_val_sets
import tensorflow as tf

import segmentation_models as sm
import cv2

image_size = (256, 256)
(train_ds, val_ds) = get_train_val_sets(1000, image_size=image_size)

model = tf.keras.models.load_model("trained.hdf5", custom_objects={
    "iou_score": sm.metrics.iou_score
})

for (image, label) in zip(val_ds.images, val_ds.labels):
    predicted = model.predict(np.array([image]))[0]
    predicted[predicted < 0.75] = 0

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label = (cv2.cvtColor(label, cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)
    predicted = (cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)

    result = cv2.hconcat((image, label, predicted))
    cv2.imshow("Test", result)
    cv2.waitKey(0)
