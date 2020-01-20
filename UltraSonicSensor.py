#https://tutorials-raspberrypi.de/entfernung-messen-mit-ultraschallsensor-hc-sr04/
import RPi.GPIO as GPIO
import time

from EmotionCalculator import EmotionCalculator


class UltraSonicSensor:

    def __init__(self, emotion_calc: EmotionCalculator):
        # GPIO Pins zuweisen
        self.GPIO_TRIGGER = 18
        self.GPIO_ECHO = 24
        self.current_distance = 0
        self.is_running = False;
        self.possible_ingredients = ["Milch", "Eier", "Brot"]

        self.emotion_calc = emotion_calc
        # GPIO Modus (BOARD / BCM)
        GPIO.setmode(GPIO.BCM)

        # Richtung der GPIO-Pins festlegen (IN / OUT)
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)

    def distance(self):
        try:
            # setze Trigger auf HIGH
            GPIO.output(self.GPIO_TRIGGER, True)

            # setze Trigger nach 0.01ms aus LOW
            time.sleep(0.00001)
            GPIO.output(self.GPIO_TRIGGER, False)

            start = time.time()
            stop = time.time()

            # speichere start
            while GPIO.input(self.GPIO_ECHO) == 0:
                start = time.time()

            # speichere Ankunftszeit
            while GPIO.input(self.GPIO_ECHO) == 1:
                stop = time.time()

            # Zeit Differenz zwischen Start und Ankunft
            time_elapsed = stop - start
            # mit der Schallgeschwindigkeit (34300 cm/s) multiplizieren
            # und durch 2 teilen, da hin und zurueck
            distance = (time_elapsed * 34300) / 2
            return distance
        except Exception as e:
            print(e)
            GPIO.cleanup()


    def is_using_fridge(self, threshold):
        self.current_distance = self.distance()
        # print("measured distance = %.1f cm" % self.current_distance)
        if self.current_distance < threshold:
            return True

        return False


