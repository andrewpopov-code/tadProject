import time

from pynput.mouse import Button, Controller

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print("clicked")

    time.sleep(10)
