import numpy as np
import pyautogui
import time
import random

from pynput import mouse, keyboard

aLower = 1.350
aUpper = 1.550
aStep = 0.05
bLower = 0.000`  #*100
bUpper = 0.012 #*100
bStep = 0.0005 #*100
cLower = 0.00000 #*1000
cUpper = 0.00004 #*1000
cStep = 0.00003 #*1000
tLower = 1
tUpper = 200
tStep = 10
aList = []
bList = []
cList = []
tList = []

for x in range (1, 1000):
    aList.append((x/1000)*(aUpper-aLower)+aLower)

for x in range (1, 1000):
    bList.append((x/1000)*(bUpper-bLower)+bLower)

for x in range (1, 1000):
    cList.append((x/1000)*(cUpper-cLower)+cLower)

for x in range (1, 1000):
    tList.append((x/1000)*(tUpper-tLower)+tLower)

"""
for x in np.arange (bLower, bUpper, bStep):
    bList.append(x)

for x in np.arange (cLower, cUpper, cStep):
    cList.append(x)

for x in np.arange (tLower, tUpper, tStep):
    tList.append(x)
"""
sets = []

for x in range (1, 1000):
    sets.append([random.choice(tList), random.choice(aList), random.choice(bList), random.choice(cList)])

def delete():
    pyautogui.press("backspace")

#while True:
 #   time.sleep(1)
   # print(pyautogui.position())
#execute()
# pyautogui.click(x, y) kliknięcie na miejsce
# time.sleep() opóźnienie
# pyautogui.typewrite() wpisanie tekstu
coords = [[]]

i = 0

def on_click(x, y, button, pressed):
    global i
    if pressed:
        print(x,y)
        coords.append([x, y])
        print(i)
        i = i + 1






mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()




while i < 14:
    time.sleep(1)

mouse_listener.stop()

for x in sets:
    pyautogui.click(coords[1][0], coords[1][1])
    time.sleep(0.1)
    pyautogui.doubleClick(coords[2][0], coords[2][1])
    delete()
    pyautogui.typewrite(str(x[0])[:6])
    pyautogui.doubleClick(coords[3][0], coords[3][1])
    delete()
    pyautogui.typewrite(str(x[1])[:6])
    pyautogui.doubleClick(coords[4][0], coords[4][1])
    delete()
    pyautogui.typewrite(str(x[2])[:6])
    pyautogui.doubleClick(coords[5][0], coords[5][1])
    delete()
    pyautogui.typewrite(str(x[3])[:10])
    pyautogui.click(coords[6][0], coords[6][1])
    pyautogui.rightClick(coords[7][0], coords[7][1])
    pyautogui.click(coords[8][0], coords[8][1])
    time.sleep(0.1)
    pyautogui.rightClick(coords[9][0], coords[9][1])
    time.sleep(0.1)
    pyautogui.click(coords[10][0], coords[10][1])
    time.sleep(0.1)
    pyautogui.click(coords[11][0], coords[11][1])
    time.sleep(0.1)
    pyautogui.click(coords[12][0], coords[12][1])
    time.sleep(0.2)
    #pyautogui.click(879, 834)
    #   time.sleep(1)
    pyautogui.typewrite(str(x[0])[:6] + "_" + str(x[1])[:6] + "_" + str(x[2])[:6] + "_" + str(x[3])[:10])
    pyautogui.click(coords[13][0], coords[13][1])
    pyautogui.click(coords[14][0], coords[14][1])

    time.sleep(0.2)