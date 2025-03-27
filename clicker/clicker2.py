import pyautogui
import time

def delete():
    for i in range(0, 100):
        pyautogui.press("backspace")

while True:
    time.sleep(1)
    print(pyautogui.position())


# pyautogui.click(x, y) kliknięcie na miejsce
# time.sleep() opóźnienie
# pyautogui.typewrite() wpisanie tekstu
