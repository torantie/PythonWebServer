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
# array position corresponds  to brightness sensor e.g. sensor 0 = possible_ingredients[0] = "Eier"
# if ingredients change, also change contents dictionary in calc_fridge_content()
possible_ingredients = ["Eier", "Milch", "Brot"]
is_observing = False


@app.route('/')
def get_data():
    contents = calc_fridge_content()
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
    contents = calc_fridge_content()
    return jsonify({'contents': contents})


def observe():
    while is_observing:
        time.sleep(1)

        if ultrasonic_sensor.is_using_fridge(15):
            print("User is in front of fridge.")
            emotion_calc.calc_and_save_emotion()
        else:
            emotion_calc.reset_predicted_emotion_dict()


def calc_fridge_content():
    occupied_sensors = brightness_sensor.get_occupied_sensors()
    contents = {'Eier': 0, 'Milch': 0, 'Brot': 0}

    if brightness_sensor.occupied_sensors_changed(occupied_sensors):
        print("Occupied sensors changed.")

        for key, value in contents.items():
            contents[key] = 0
            # print("In reset:" + str(key) + " "+ str(contents[key]))

        for i, is_sensor_occupied in enumerate(occupied_sensors):
            if is_sensor_occupied:
                # print("sensor " + str(i) + " is occupied. Add 1 to " + possible_ingredients[i])
                contents[possible_ingredients[i]] += 1

                print("fridge has " + str(contents[possible_ingredients[i]]) + " of ingredient "
                      + possible_ingredients[i])

    else:
        print("Occupied sensors did not change.")

    brightness_sensor.set_last_known_occupied_sensors(occupied_sensors)
    return contents


def start_observing():
    global is_observing
    is_observing = True
    t = threading.Thread(target=observe)
    t.start()


def stop_observing():
    global is_observing
    is_observing = False




# threaded false aufgrund eines fehlers in keras (https://github.com/keras-team/keras/issues/13353)
# könnte auch anderen fix geben
if __name__ == '__main__':
    start_observing()
    app.run(host="0.0.0.0", port=5000, threaded=False)

