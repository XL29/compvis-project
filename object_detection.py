import csv
from djitellopy import Tello
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


model_path = './model/object_classifier/efficientdet_lite0.tflite'

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(image,detection_result) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv.putText(image, result_text, text_location, cv.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#with ObjectDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.

detector = vision.ObjectDetector.create_from_options(options)


def objectDetection(frame):
  rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  detection_result = detector.detect(mp_image) 
  image_copy = np.copy(mp_image.numpy_view())

  return detection_result
