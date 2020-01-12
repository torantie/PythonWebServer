from picamera import PiCamera
from time import sleep

import io
import time
import cv2
import numpy as np
import sys


# https://picamera.readthedocs.io/en/release-1.10/recipes1.html#capturing-to-a-pil-image
def take_pictures(n, delay):
    try:
        images = []
        camera = PiCamera()

        camera.start_preview()

        for i in n:
            # Create the in-memory stream
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            time.sleep(delay)
            # Construct a numpy array from the stream
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            # "Decode" the image from the array, preserving colour
            image = cv2.imdecode(data, 1)
            # OpenCV returns an array with data in BGR order. If you want RGB instead
            # use the following...
            image = image[:, :, ::-1]
            images.append(image)
    finally:
        camera.close()
        return images


