import Adafruit_ADS1x15


class BrightnessSensor:
    # sensor pins range(1, 4)
    def __init__(self, number_of_sensors, sensor_pins, threshold):
        self.GAIN = 1
        self.adc = Adafruit_ADS1x15.ADS1115()
        self.number_of_sensors = number_of_sensors
        self.sensor_pins = sensor_pins
        self.occupied_threshold = threshold
        # maybe check for sensor pins == number of sensors

    def get_occupied_sensors(self):
        occupied_sensors = [False]*self.number_of_sensors

        for i, pin in enumerate(self.sensor_pins):
            print("pin number " + str(pin) + " saved in occupied sensors array with index " + str(i))
            read_value = self.adc.read_adc(pin, gain=self.GAIN)
            print("read value: " + str(read_value) + " occupied_threshold: " + str(self.occupied_threshold))
            if self.occupied_threshold <= read_value:
                occupied_sensors[i] = True
            else:
                occupied_sensors[i] = False

        return occupied_sensors

