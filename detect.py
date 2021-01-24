import time

import cv2
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string("framework", "tf", "(tf, tflite, trt")
flags.DEFINE_string("weights", "./checkpoints/yolov4-416", "path to weights file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_string("image", "./data/kite.jpg", "path to input image")
flags.DEFINE_string("output", "result.png", "path to output image")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.25, "score threshold")


def main(_argv):
    input_size = FLAGS.size
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.0

    images_data = []
    images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    saved_model_loaded = tf.saved_model.load(
        FLAGS.weights, tags=[tag_constants.SERVING]
    )

    start_time = time.time()

    infer = saved_model_loaded.signatures["serving_default"]
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score,
    )

    end_time = time.time()

    image_h, image_w, c = original_image.shape
    coor = boxes.numpy()[0][0]
    if coor is not None and len(coor) == 4:
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        img_crop = original_image[
            int(coor[0]) : int(coor[2]), int(coor[1]) : int(coor[3])
        ]
        cv2.imwrite("test.png", img_crop)

    print("inference [secs]: %s" % (end_time - start_time))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
