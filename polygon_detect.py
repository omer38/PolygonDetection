import cv2 as cv
import numpy as np

#OpenCv Fonts
font1 = cv.FONT_HERSHEY_SIMPLEX
font2 = cv.FONT_HERSHEY_COMPLEX

#First convert to gray then convert to binary
img = cv.imread("polygons.png")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
_,threshold = cv.threshold(gray,240,255,cv.THRESH_BINARY)

#Find countours
countours,_=cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#Approximate the countour value and then draw countours
for cnt in countours:
    epsilon = 0.01*cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon,True)

    cv.drawContours(img,[approx],0,(0),5)

    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx)==3:
        cv.putText(img,"Triangle",(x,y),font1,1,(0))
    elif len(approx) == 4:
        cv.putText(img, "Rectangle", (x, y), font1,1, (0))
    elif len(approx) == 5:
        cv.putText(img, "Pentagon", (x, y), font1,1, (0))
    elif len(approx) == 6:
        cv.putText(img, "Hegzagon", (x, y), font1,1, (0))
    else:
        cv.putText(img, "Elips", (x, y), font1,1, (0))

cv.imshow("IMG",img)
cv.waitKey(0)
cv.destroyAllWindows()
