import cv2
from  imutils import contours
from  imutils.perspective import four_point_transform
import imutils as im
import numpy as np
import math

def digitalReader(Img):

        #matrix for defining the digits based on the layout of the graphic
    DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
    }
    vals=sqFinder(Img)
    edged=vals[1]
    warped=vals[0]
    clean=vals[0]
    #clean=cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    lowColor=(10,120,120)
    #lowColor=(238,128,18)
    highColor=(255,255,255)
    #cv2.imshow('Gauge Picture', edged)
    #cv2.waitKey(0)
    warped=cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(warped,lowColor,highColor )
    warped=cv2.bitwise_and(clean,clean,mask=mask)
    warpedCut=cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warpedCut=cv2.GaussianBlur(warpedCut,(5,5),2)
    warpedCut=cv2.threshold(warpedCut,70,255,cv2.THRESH_BINARY)[1]
    warped=cv2.GaussianBlur(warped,(5,5),5)
    #warped=cv2.GaussianBlur(warped,(5,5),1)
    #cv2.imshow('Gauge Picture1', warpedCut)
    #cv2.waitKey(0)
    warped=cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  #convert to blur
    warped=cv2.threshold(warped,0,255,cv2.THRESH_BINARY)[1]
    

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    cnts=cv2.findContours(warpedCut.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts=im.grab_contours(cnts)
    remove=[]
    mask = np.ones(warpedCut.shape[:2], dtype="uint8") * 255
    for c in cnts:
        if cv2.contourArea(c)<80: #if the area is sufficently small then we have an important item
            remove.append(c)
            cv2.drawContours(mask,[c],-1,0,6)
    clean2=clean.copy()
    #cv2.drawContours(clean2, remove, -1, (255,255,0), 8)
    #cv2.imshow('Gauge ', clean2)
    #cv2.imshow('warped1 ', warpedCut)
    warped = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)
    warped=cv2.bitwise_and(warped,warped,mask=mask)
    #cv2.imshow('Gauge Picture2', warped)
    #cv2.waitKey(0)

    cnts=cv2.findContours(warped.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=im.grab_contours(cnts)
    cv2.drawContours(warped, cnts, -1, (0,255,0), 1)
    cv2.imshow('Gauge Picture', warped)
    cv2.waitKey(0)
    temp=[]
    for c in cnts:
        if cv2.contourArea(c)>800: #if the area is sufficently large then we have an important item
            cow=cv2.boundingRect(c)
            temp.append(c)
    cv2.drawContours(clean, temp, -1, (255,255,0), 1)
    #cv2.imshow('Gauge Picture3', clean)
    #cv2.waitKey(0)
    digit=contours.sort_contours(temp, method="left-to-right")[0]
    digits=[]
    w_max=0
    [x,y,w,h]=cv2.boundingRect(digit[0])
    #w_max=w
    w_max=55
    #print(digit)
    #print([x,y,w,h], len(digit))
    origin=x
    #for d in digit:
    i=0
    for d in digit:
        [a,y,c,h]=cv2.boundingRect(d)
        x=origin+w_max*i
        #cv2.rectangle(clean,(x,y),(x+w_max,y+h),(0,50*i,0),2)
        #cv2.rectangle(clean,(x+w-w_max,y),(x+w,y+h),(0,255,0),2)
        #print(d)
        #roi = warped[y:y+h,x:x+w_max]
        roi = warped[y:y+h,x+w-w_max:x+w]#gets the rectangle around the digits
        #now to break the rectangle into smaller sections for comparing to the digits index
        #cv2.drawContours(clean,rect,-1,0,1)
        #print(roi[0])
        x=x+w-w_max
        dW=int(.175*roi.shape[0])
        #print(dW)
        dH=int(0.15*roi.shape[1])
        breakout=[  
            ((x,y),(x+w_max,y+2*dH)), #top
            ((x,y),(x+dW,y+h//2)), #top left
            ((x+w_max-dW,y),(x+w_max,y+h//2)), #top right 
            ((x,y+h//2-dH//2),(x+w_max,y+h//2+dH//2)), #middle
            ((x,y+h//2),(x+dW,y+h)), #bottom left
            ((x+w_max-dW,y+h//2),(x+w_max,y+h)), #bottom right
            ((x,y+h-2*dH),(x+w_max,y+h)), #bottom
            ]
        light = len(breakout)*[0] #makes a tuple of arrays of zeros
        #print(light)
        i=i+1
        #print
        
        for (e,((x1,y1),(x2,y2))) in enumerate(breakout):
                #print(e)
                #print(((x1,y1),(x2,y2)))
                cv2.rectangle(clean,(x1,y1),(x2,y2),(50*i,50*i,0),5-i)
                sec=warped[y1:y2,x1:x2]
                t=cv2.countNonZero(sec)
                #print(t)
                A=(x2-x1)*(y2-y1)
                if(t/A)>.5:
                    light[e]=1
        digit=DIGITS_LOOKUP[tuple(light)]
        digits.append(digit)
        #print(light)
        #    sum=cv2.countNonZero(sec)

    #print(digits)
    output=""
    for d in digits:
        output=output+str(d)
    output=output[:-2]+"."+output[-2:]+'\N{DEGREE SIGN}'+'C'
    print(output)
    cv2.imshow('Gauge', warped)
    cv2.imshow('Gauge Pic', clean)
    cv2.waitKey(0)

    return 0


def sqFinder(Img):
    gray=cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)  #convert to blur
    blur=cv2.GaussianBlur(gray,(5,5),0)         #blur the image
    thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1] #if pixel does not meet threshold sets the value to zero
    edged=cv2.Canny(blur, 50, 200, 255)
    #this image provides enough contrast between the thermometer and backgroud that we can determine it from the other parts of the image
    
    cnts= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=im.grab_contours(cnts)     #makes an array of all the contours from the edged image
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) #sorts the array of contours

    cv2.imshow('Gauge Picture3', edged)
    cv2.waitKey(0)

    for c in cnts:
        p=cv2.arcLength(c,True) #finds the perimeter for a closed shape , true=close shape
        epsilon=.1*p   #sets the sensityivity of the contour lines.
        approx=cv2.approxPolyDP(c,epsilon,True)

        if len(approx)==4:
            #if we find a square (4 corners) return that
            cnt=approx
            break

    warped=four_point_transform(Img,cnt.reshape(4,2)) #change the shape of the orginal edged matrix to fit the new desired image bounds
    edged=four_point_transform(edged,cnt.reshape(4,2))
    warped=im.resize(warped, width=425)
    val=[warped,edged]

    cv2.imshow('Gauge Picture3', warped)
    cv2.waitKey(0)

    return val

def main(path):
    #gaugeImg = cv2.imread('C:\\Users\\cftra\OneDrive\\Desktop\\ME_459\\Project\\IMG_3672.JPEG')
    digitalImg = cv2.imread(path)
    #digitalImg=im.resize(digitalImg, width=2500)
    cv2.imshow('Gauge Picture3', digitalImg)
    cv2.waitKey(0)
    digitalReader(digitalImg)
    return 0

main('C:\\Users\\cftra\OneDrive\\Desktop\\ME_459\\FinalProject\\ME459FinalProject\\IMG_3681_4.JPEG')
