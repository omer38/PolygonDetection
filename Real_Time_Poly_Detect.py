import cv2 as cv
import numpy as np

font = cv.FONT_HERSHEY_SIMPLEX

def nothing(x):
    """
    Define this function for using of trackbar
    """
    pass

cap = cv.VideoCapture(0)
cv.namedWindow("Settings")

cv.createTrackbar("Lower-Hue","Settings",0,180,nothing)
cv.createTrackbar("Lower-Saturation","Settings",0,255,nothing)
cv.createTrackbar("Lower-Value","Settings",0,255,nothing)
cv.createTrackbar("Upper-Hue","Settings",0,180,nothing)
cv.createTrackbar("Upper-Saturation","Settings",0,255,nothing)
cv.createTrackbar("Upper-Value","Settings",0,255,nothing)

while True:
    ret,frame = cap.read()
    frame = cv.flip(frame,1)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lh = cv.getTrackbarPos("Lower-Hue","Settings")
    ls = cv.getTrackbarPos("Lower-Saturation", "Settings")
    lv = cv.getTrackbarPos("Lower-Value", "Settings")
    uh = cv.getTrackbarPos("Upper-Hue", "Settings")
    us = cv.getTrackbarPos("Upper-Saturation", "Settings")
    uv = cv.getTrackbarPos("Upper-Value", "Settings")

    lower_color = np.array([lh,ls,lv])
    upper_color = np.array([uh,us,uv])

    mask = cv.inRange(hsv,lower_color,upper_color)
    kernel = np.ones((5,5),np.uint8)
    mask = cv.erode(mask,kernel)

    contours,_ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        epsilon = 0.02*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)

        x = approx.ravel()[0]
        y = approx.ravel()[1]#Where contours starts
        if area > 1000:
            cv.drawContours(frame,[approx],0,(0,0,0),5)

        if len(approx) == 3:
            cv.putText(frame,"Triangle",(x,y),font,1,(0,0,0))
        elif len(approx) == 3:
            cv.putText(frame,"Rectangle",(x,y),font,1,(0,0,0))
        elif len(approx) > 6:
            cv.putText(frame,"Circle",(x,y),font,1,(0,0,0))

    cv.imshow("Webcam",frame)
    cv.imshow("Mask",mask)

    if cv.waitKey(3) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
