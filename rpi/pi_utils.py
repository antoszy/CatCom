import RPi.GPIO as GPIO


class Button:
    def __init__(self, gpio_num):
        self.gpio_num = gpio_num
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_num, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def check_value(self):
        return not GPIO.input(self.gpio_num)

