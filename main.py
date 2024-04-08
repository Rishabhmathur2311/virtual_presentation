import cv2
import os
import mediapipe as mp
import numpy as np
import math

class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        return allHands, img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmlist

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


#variables
width=1280
height=720

folderPath="images"

imgNumber=0
hs, ws=120, 213

gestureThreshold=300

buttonPressed=False
buttonCounter=0
buttonDelay=15

annotations=[[]]
annotationNumber=-1
annotationStart=False

#camera setup
cap=cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)

pathImages=sorted(os.listdir(folderPath), key=len)
print(pathImages)

detector=handDetector(maxHands=1)

while True:

    success, img=cap.read()

    img=cv2.flip(img, 1)

    pathFullImage=os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent=cv2.imread(pathFullImage)

    #overlapping the two images
    imgSmall=cv2.resize(img, (ws, hs))
    h, w, _=imgCurrent.shape
    imgCurrent[0:hs, w-ws:w]=imgSmall

    hands, img=detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed==False:
        hand=hands[0]
        fingers=detector.fingersUp(hand)
        print(fingers)

        cx, cy=hand['center']
        lmList=hand['lmList']

        xval=int(np.interp(lmList[8][0], [width//2, w], [0, width]))
        yval = int(np.interp(lmList[8][1], [150, height], [0, height]))

        indexFinger = xval, yval

        if cy<=gestureThreshold:

            # 1st gesture
            if fingers == [1, 0, 0, 0, 0]:
                print('left')
                if imgNumber>0:
                    imgNumber-=1
                    buttonPressed=True
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False

            # 2nd gesture
            if fingers == [0, 0, 0, 0, 1]:
                print('right')
                if imgNumber<len(pathImages)-1:
                    imgNumber+=1
                    buttonPressed=True
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False

        # 3rd gesture
        if fingers == [0, 1, 0, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0 ,255), cv2.FILLED)
            annotationStart=False


        if fingers == [0, 1, 1, 0, 0]:
            if annotationStart is False:
                annotationStart=True
                annotationNumber+=1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0 ,255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart=False

        if fingers == [0, 1, 1, 1, 1]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart=False


    #iterations for button pressed

    if buttonPressed:
        buttonCounter+=1
        if buttonCounter>buttonDelay:
            buttonCounter=0
            buttonPressed=False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j > 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12)




    cv2.imshow("img", img)
    cv2.imshow("presentation", imgCurrent)
    key=cv2.waitKey(1)

    if key==ord('q'):
        break

