from builtins import dict, zip, len, print

import cv2
import face_recognition
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from timeit import default_timer as timer
from PiCam import take_pictures


class EmotionCalculator:
    current_emotion = "Happy"
    # Model from https://github.com/omar178/Emotion-recognition.git
    model = load_model("./models/_mini_XCEPTION.102-0.66.hdf5")
    model_dim = (64, 64)
    emotion_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

    label_map = dict((v, k) for k, v in emotion_dict.items())

    def calc_and_save_emotion(self):
        self.current_emotion = self.calculate_emotion_pi_cam()

    def calculate_emotion_with_window(self):
        start = timer()
        predicted_emotions_dictionary = {}

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

                    if predicted_label in predicted_emotions_dictionary:
                        predicted_emotions_dictionary[predicted_label] += 1
                    else:
                        predicted_emotions_dictionary[predicted_label] = 0

            # if there are no faces write 'no face found' on the image and display
            else:
                cv2.putText(img, "No face found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow('Emotion', img)

            if cv2.waitKey(1) == 13:  # enter key to end program
                return ""

        # disconnect from the webcam and close the image
        cap.release()
        cv2.destroyWindow('Emotion')

        return self.get_max_occurence_emotion(predicted_emotions_dictionary)

    def calculate_emotion_pi_cam(self):
        predicted_emotions_dictionary = {}
        pictures = take_pictures(5, 0)

        for picture in pictures:
            rects, faces = self.face_detector(picture)
            self.calc_predicted_emotions(faces, predicted_emotions_dictionary)

        return self.get_max_occurence_emotion(predicted_emotions_dictionary)

    def calculate_emotion_web_cam(self):
        start = timer()
        predicted_emotions_dictionary = {}
        cap = cv2.VideoCapture(0)

        while timer() - start < 5:
            print(timer() - start)

            ret, img = cap.read()
            rects, faces = self.face_detector(img)

            self.calc_predicted_emotions(faces, predicted_emotions_dictionary)

            if cv2.waitKey(1) == 13:  # enter key to end program
                return ""

        cap.release()
        return self.get_max_occurence_emotion(predicted_emotions_dictionary)

    def calc_predicted_emotions(self, faces, predicted_emotions_dictionary):
        if np.sum([faces[0]]) != 0.0:
            for face in faces:
                predicted_class = np.argmax(self.model.predict(face))
                predicted_label = self.label_map[predicted_class]

                if predicted_label in predicted_emotions_dictionary:
                    predicted_emotions_dictionary[predicted_label] += 1
                else:
                    predicted_emotions_dictionary[predicted_label] = 0

    def get_max_occurence_emotion(self, predicted_emotions_dictionary):
        max_occurrence_emotion = ""
        max_count = 0
        for predicted_emotion in predicted_emotions_dictionary:
            if max_count < predicted_emotions_dictionary[predicted_emotion]:
                max_occurrence_emotion = predicted_emotion
                max_count = predicted_emotions_dictionary[predicted_emotion]

        print("max_occurrence_emotion: " + max_occurrence_emotion)
        return max_occurrence_emotion

    # face detection function
    def face_detector(self, img):
        # convert image to grayscale and find faces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_recognition.face_locations(img_gray)

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

            # resize the face image and preprocess to fit the model requirements
            face = img_gray[top:bottom, left:right]
            face = cv2.resize(face, self.model_dim, interpolation=cv2.INTER_AREA)
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face_img.append(face)

        return rect, face_img
