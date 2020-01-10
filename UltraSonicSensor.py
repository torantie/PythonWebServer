#https://tutorials-raspberrypi.de/entfernung-messen-mit-ultraschallsensor-hc-sr04/
import RPi.GPIO as GPIO
import time
import threading

from EmotionCalculator import EmotionCalculator


class UltraSonicSensor:
    # GPIO Pins zuweisen
    __GPIO_TRIGGER = 18
    __GPIO_ECHO = 24
    __current_distance = 0

    def __init__(self, emotion_calc:EmotionCalculator):
        self.emotion_calc = emotion_calc
        # GPIO Modus (BOARD / BCM)
        GPIO.setmode(GPIO.BCM)

        # Richtung der GPIO-Pins festlegen (IN / OUT)
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)

    def __distance(self):
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

    def is_using_fridge(self, threshold):
        if self.current_distance < threshold:
            return True

        return False

    def __observe(self):
        try:
            while True:
                current_distance = self.distance()
                print ("measured distance = %.1f cm" % current_distance)
                time.sleep(1)

                if self.is_using_fridge(5):
                    self.emotion_calc.calc_and_save_emotion()

            # Beim Abbruch durch STRG+C resetten
        except KeyboardInterrupt:
            print("Measurement interrupted.")
            GPIO.cleanup()

    def start(self):
        t = threading.Thread(target=self.observe())
        t.start()
