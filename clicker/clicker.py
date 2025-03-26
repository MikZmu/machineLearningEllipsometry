
import keyboard
import pyautogui
import time

import pynput
from pynput import mouse, keyboard
import numpy as np

aLower = 10
aUpper = 100
aStep = 1
bLower = 1 #*100
bUpper = 2 #*100
bStep = 0.1 #*100
cLower = 0.001 #*1000
cUpper = 0.002 #*1000
cStep = 0.0001 #*1000
tLower = 1
tUpper = 100
tStep = 5
aList = []
bList = []
cList = []
tList = []


for x in np.arange (aLower, aUpper, aStep):
    aList.append(x)

for x in np.arange (bLower, bUpper, bStep):
    bList.append(x)

for x in np.arange (cLower, cUpper, cStep):
    cList.append(x)

for x in np.arange (tLower, tUpper, tStep):
    tList.append(x)



# List to store recorded actions
actions = []

# Flag to stop recording
recording = True

# Adjustable delay between events
delay = 0.25  # Adjust this value as needed

def on_click(x, y, button, pressed):
    if recording:
        actions.append(('mouse_click', x, y, button, pressed, time.time()))


def on_press(key):
    global recording
    if key == keyboard.Key.esc:
        # Stop recording on 'esc' key press
        recording = False
        print("Stop recording")
        mouse_listener.stop()
        keyboard_listener.stop()
    if recording:
        actions.append(('key_press', key, time.time()))

def on_release(key):
    if recording:
        actions.append(('key_release', key, time.time()))

def execute():
    for A in aList:
        for B in bList:
            for C in cList:
                for T in tList:
                    replay_actions(A, B, C, T)
                    time.sleep(0.5)  # Adjust the delay as needed

def replay_actions(A,B,C,T):
    print("replaying actions")
    start_time = actions[0][-1]
    for action in actions:
        event_time = action[-1]
        time.sleep(event_time - start_time + delay)
        start_time = event_time
        if action[0] == 'mouse_click':
            _, x, y, button, pressed, _ = action
            if pressed:
                pyautogui.mouseDown(x, y, button=button.name)
            else:
                pyautogui.mouseUp(x, y, button=button.name)
        elif action[0] == 'key_press':
            _, key, _ = action
            pyautogui.keyDown(key.char if hasattr(key, 'char') else key.name)
            if key == keyboard.Key.A:
                pyautogui.typewrite(str(A))
            if key == keyboard.Key.B:
                pyautogui.typewrite(str(B))
            if key == keyboard.Key.C:
                pyautogui.typewrite(str(C))
            if key == keyboard.Key.T:
                pyautogui.typewrite(str(T))
            if key == keyboard.Key.n:
                pyautogui.typewrite(str(T) + "_" + str(A) + "_" + str(B*100) + "_" + str(C*1000))


        elif action[0] == 'key_release':
            _, key, _ = action
            pyautogui.keyUp(key.char if hasattr(key, 'char') else key.name)

while recording:
    with mouse.Listener(on_click=on_click) as mouse_listener, keyboard.Listener(on_press=on_press, on_release=on_release) as keyboard_listener:
        print("Recording actions...")
        mouse_listener.join()
        keyboard_listener.join()
        print("Recording completed")
        if(recording == False):
            break

print("check")

execute()
print("check2")
# Replay actions until a certain combination of buttons is pressed

