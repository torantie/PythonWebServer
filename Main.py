from flask import Flask
from flask import jsonify
import os

from EmotionCalculator import EmotionCalculator
# from UltraSonicSensor import UltraSonicSensor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
emotion_calc = EmotionCalculator()
# sensor = UltraSonicSensor(emotion_calc)
contents = {'Eier': 1, 'Milch': 3}


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


# threaded false aufgrund eines fehlers in keras (https://github.com/keras-team/keras/issues/13353)
# könnte auch anderen fix geben
if __name__ == '__main__':
    # sensor.start()
    app.run(host="0.0.0.0", port=5000, threaded=False)

