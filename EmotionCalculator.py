from builtins import dict, zip, len, print

import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from timeit import default_timer as timer
from PiCam import take_pictures


class EmotionCalculator:
    def __init__(self):
        self.current_emotion = "Happy"
        # Model from https://github.com/omar178/Emotion-recognition.git
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.model = load_model("./models/_mini_XCEPTION.102-0.66.hdf5")
        self.model._make_predict_function()

        self.model_dim = (64, 64)
        self.emotion_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
        self.label_map = dict((v, k) for k, v in self.emotion_dict.items())
        self.predicted_emotions_dictionary = {}

    def calc_and_save_emotion(self):
        self.calculate_emotion_pi_cam()
        self.current_emotion = self.get_max_occurence_emotion()

    def reset_predicted_emotion_dict(self):
        self.predicted_emotions_dictionary.clear()

    def calculate_emotion_with_window(self):
        start = timer()

        # start webcam capture and continuous loop
        cap = cv2.VideoCapture(0)
        while timer() - start < 5:
            print(timer() - start)
            # read a frame from the webcam and find faces
            ret, img = cap.read()
            rects, faces = self.face_detector(img)

            # draw boxes around all the faces and write the emotions
            if np.sum([faces[0]]) != 0.0:
                for (top, right, bottom, left), face in zip(rects, faces):
                    # find the prediction form the ML model and associated label
                    predicted_class = np.argmax(self.model.predict(face))
                    predicted_label = self.label_map[predicted_class]
                    # position for text near the bottom left corner of the face
                    label_pos = (left - 15, bottom + 25)
                    # write the label on the image
                    cv2.putText(img, predicted_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    if predicted_label in self.predicted_emotions_dictionary:
                        self.predicted_emotions_dictionary[predicted_label] += 1
                    else:
                        self.predicted_emotions_dictionary[predicted_label] = 0

            # if there are no faces write 'no face found' on the image and display
            else:
                cv2.putText(img, "No face found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow('Emotion', img)

            if cv2.waitKey(1) == 13:  # enter key to end program
                return ""

        # disconnect from the webcam and close the image
        cap.release()
        cv2.destroyWindow('Emotion')

    def calculate_emotion_pi_cam(self):
        pictures = take_pictures(2, 0.001)

        for picture in pictures:
            rects, faces = self.face_detector(picture)
            self.calc_predicted_emotions(faces)

    def calculate_emotion_web_cam(self):
        start = timer()
        cap = cv2.VideoCapture(0)

        while timer() - start < 5:
            print(timer() - start)

            ret, img = cap.read()
            rects, faces = self.face_detector(img)

            self.calc_predicted_emotions(faces)

            if cv2.waitKey(1) == 13:  # enter key to end program
                return ""

        cap.release()

    def calc_predicted_emotions(self, faces):
        if np.sum([faces[0]]) != 0.0:
            for face in faces:
                with self.graph.as_default():
                    set_session(self.sess)
                    predicted_class = np.argmax(self.model.predict(face))
                    predicted_label = self.label_map[predicted_class]

                    if predicted_label in self.predicted_emotions_dictionary:
                        self.predicted_emotions_dictionary[predicted_label] += 1
                    else:
                        self.predicted_emotions_dictionary[predicted_label] = 0

    def get_max_occurence_emotion(self):
        max_occurrence_emotion = ""
        max_count = 0
        for predicted_emotion in self.predicted_emotions_dictionary:
            if max_count < self.predicted_emotions_dictionary[predicted_emotion]:
                max_occurrence_emotion = predicted_emotion
                max_count = self.predicted_emotions_dictionary[predicted_emotion]

        print("max_occurrence_emotion: " + max_occurrence_emotion)
        return max_occurrence_emotion

    # face detection function
    def face_detector(self, img):
        # convert image to grayscale and find faces
        # print("convert image to grayscale and find faces")
        # scale picture
        small_image = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
        img_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        # print("after grayscale, before faces")
        faces = face_recognition.face_locations(img_gray,1,"hog")
        # print("converted image to grayscale and find faces")
        # return zeros if there are no faces detected
        if len(faces) == 0:
            return (0, 0, 0, 0), np.zeros(self.model_dim, np.uint8)

        # otherwise return any faces detected
        # initiate lists in case there are multiple faces
        rect = []
        face_img = []

        # if faces are found, draw a rectangle and cut out just the face
        for (top, right, bottom, left) in faces:
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            rect.append((top, right, bottom, left))
            # print("left: "+str(left)+ " top: " + str(top) + " right: " + str(right) +" bottom: " + str(bottom))
            # resize the face image and preprocess to fit the model requirements
            # print("before resize the face image and preprocess to fit the model requirements")
            face = img_gray[top:bottom, left:right]
            face = cv2.resize(face, self.model_dim, interpolation=cv2.INTER_AREA)
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face_img.append(face)
            # print("after resize the face image and preprocess to fit the model requirements")

        return rect, face_img
