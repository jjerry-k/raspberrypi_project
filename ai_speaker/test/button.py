import RPi.GPIO as GPIO
import time

BUTTON = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN)

while True:
    state = GPIO.input(BUTTON)
    if state:
        pass
    else:
        print("on")
