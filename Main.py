from flask import Flask
from flask import jsonify
import os
import threading
import sys
import time

from BrightnessSensor import BrightnessSensor
from EmotionCalculator import EmotionCalculator
from UltraSonicSensor import UltraSonicSensor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
emotion_calc = EmotionCalculator()
ultrasonic_sensor = UltraSonicSensor(emotion_calc)
# 3 sensors; pins 1,2,3; threshold 12000
brightness_sensor = BrightnessSensor(3, range(1, 4), 12000)
contents = {'Eier': 1, 'Milch': 3, 'Brot': 0}
# array position corresponds  to brightness sensor e.g. sensor 0 = possible_ingredients[0] = "Eier"
possible_ingredients = ["Eier", "Milch", "Brot"]
is_observing = False


@app.route('/')
def get_data():
    return jsonify({'emotion': emotion_calc.current_emotion, 'contents': contents});


@app.route('/emotion')
def get_emotion_data():
    return jsonify({'emotion': emotion_calc.current_emotion})


@app.route('/calcEmotion')
def calc_emotion_data():
    emotion_calc.calc_and_save_emotion()
    return jsonify({'emotion': emotion_calc.current_emotion})


@app.route('/contents')
def get_locations_data():
    return jsonify({'contents': contents})


def observe():
    while is_observing:
        time.sleep(1)

        if ultrasonic_sensor.is_using_fridge(15):
            emotion_calc.calc_and_save_emotion()
            for i, is_sensor_occupied in enumerate(brightness_sensor.get_occupied_sensors()):
                if is_sensor_occupied:
                    print("sensor " + str(i) + " is occupied. Add 1 to " + possible_ingredients[i])
                    contents[possible_ingredients[i]] += 1
                    print("fridge has " + str(contents[possible_ingredients[i]]) + " of ingredient "
                          + possible_ingredients[i])


def start_observing():
    global is_observing
    is_observing = True
    t = threading.Thread(target=observe())
    t.start()


def stop_observing():
    global is_observing
    is_observing = False


# threaded false aufgrund eines fehlers in keras (https://github.com/keras-team/keras/issues/13353)
# könnte auch anderen fix geben
if __name__ == '__main__':
    start_observing()
    app.run(host="0.0.0.0", port=5000, threaded=False)

