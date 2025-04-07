import numpy as np
import pyautogui
import time
import random

aLower = 1.350
aUpper = 1.550
aStep = 0.05
bLower = 0.008 #*100
bUpper = 0.012 #*100
bStep = 0.0005 #*100
cLower = 0.00001 #*1000
cUpper = 0.00002 #*1000
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
                    pyautogui.rightClick(62, 561)
                    time.sleep(1)
                    pyautogui.click(178, 647)
                    time.sleep(1)
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

#execute()
# pyautogui.click(x, y) kliknięcie na miejsce
# time.sleep() opóźnienie
# pyautogui.typewrite() wpisanie tekstu


for x in sets:
    pyautogui.click(105, 173)
    time.sleep(0.1)
    pyautogui.doubleClick(537, 482)
    delete()
    pyautogui.typewrite(str(x[0])[:6])
    pyautogui.doubleClick(564, 547)
    delete()
    pyautogui.typewrite(str(x[1])[:6])
    pyautogui.doubleClick(565, 635)
    delete()
    pyautogui.typewrite(str(x[2])[:6])
    pyautogui.doubleClick(766, 491)
    delete()
    pyautogui.typewrite(str(x[3])[:6])
    pyautogui.click(301, 658)
    pyautogui.rightClick(62, 561)
    pyautogui.click(178, 647)
    time.sleep(0.1)
    pyautogui.rightClick(730  , 617)
    time.sleep(0.1)
    pyautogui.click(849, 759)
    time.sleep(0.1)
    pyautogui.click(1270, 851)
    time.sleep(0.1)
    pyautogui.click(833, 889)
    time.sleep(0.2)
    #pyautogui.click(879, 834)
    #   time.sleep(1)
    pyautogui.typewrite(str(x[0])[:6] + "_" + str(x[1])[:6] + "_" + str(x[2])[:6] + "_" + str(x[3])[:6])
    pyautogui.click(936, 786)
    pyautogui.click(1167, 913)

    time.sleep(0.2)