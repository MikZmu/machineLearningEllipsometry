import numpy as np
import pyautogui
import time

aLower = 1.305
aUpper = 1.595
aStep = 0.05
bLower = 0.009 #*100
bUpper = 0.011 #*100
bStep = 0.0005 #*100
cLower = 0.0001 #*1000
cUpper = 0.00013 #*1000
cStep = 0.00003 #*1000
tLower = 1
tUpper = 200
tStep = 10
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

def delete():
    pyautogui.press("backspace")

#while True:
 #   time.sleep(1)
   # print(pyautogui.position())


def execute():
    total = len(aList) * len(bList) * len(cList) * len(tList)
    progress = 0
    for A in aList:
        for B in bList:
                for T in tList:
                    pyautogui.click(68, 163)
                    time.sleep(0.1)
                    pyautogui.doubleClick(909,485)
                    delete()
                    pyautogui.typewrite(str(T))
                    pyautogui.doubleClick(940, 550)
                    delete()
                    pyautogui.typewrite(str(A))
                    pyautogui.doubleClick(940, 592)
                    delete()
                    pyautogui.typewrite(str(B))
                    pyautogui.doubleClick(940, 634)
                    delete()
                    pyautogui.typewrite(str(0.0001))
                    pyautogui.click(1147,489)
                    pyautogui.rightClick(331, 587)
                    pyautogui.click(442, 671)
                    time.sleep(0.1)
                    pyautogui.rightClick(1456,814)
                    time.sleep(0.1)
                    pyautogui.click(1641, 656)
                    time.sleep(0.1)
                    pyautogui.click(1207, 758)
                    time.sleep(0.1)
                    pyautogui.click(659, 788)
                    time.sleep(0.1)
                    pyautogui.click(622, 784)
                    time.sleep(0.1)
                    pyautogui.typewrite(str(T) + "_" + str(A) + "_" + str(B)[:6] + "_" + str(0.0001)[:8])
                    pyautogui.click(646, 745)
                    pyautogui.click(1171, 869)

                    time.sleep(0.2)  # Adjust the delay as needed
                    #progress = A * B * C * T
                    #print("Progress: {}/{}".format(progress, total))
                    #print("A: {}, B: {}, C: {}, T: {}".format(A, B, C, T))

execute()
# pyautogui.click(x, y) kliknięcie na miejsce
# time.sleep() opóźnienie
# pyautogui.typewrite() wpisanie tekstu
