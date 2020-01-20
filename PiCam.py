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

        for i in range(0, n):
            # Create the in-memory stream
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            # Construct a numpy array from the stream
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            # "Decode" the image from the array, preserving colour
            image = cv2.imdecode(data, 1)

            images.append(image)
            time.sleep(delay)


    except Exception as e:
        print(e)
    finally:
        camera.close()
        return np.array(images)


