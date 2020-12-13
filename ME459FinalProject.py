#Code adapted from the Medium article (for the analog gauge) by Abhijeet Nayak at
#https://medium.com/@nayak.abhijeet1/analogue-gauge-reader-using-computer-vision-62fbd6ec84cc

#Code refered to from OpenCV2 python documentation and Adrian Rosebrock's Recognizing digits with OpenCV and Python
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_table_of_contents_gui/py_table_of_contents_gui.html
#https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

#Code developed for the UW-Madison MINLAB by Mark Vandenberg and Collin Trafton
#Code submitted for ME459 Final Project, Fall 2020



import cv2
from  imutils import contours
from  imutils.perspective import four_point_transform
import imutils as im
import numpy as np
import math
import os

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
    vals=sqFinder(Img)  #see sqFinder
    edged=vals[1]   #gets the edged copy from the sqFinder
    warped=vals[0]  #gets the warped orginal image from sqFinder
    clean=vals[0]   #gets a second warped copy from sqFinder
    try:        
        #clean=cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        lowColor=(10,120,120)       #lower bound on color isolation

        highColor=(255,255,255)     #upper bound on color isolation
        #cv2.imshow('Gauge Picture', edged)
        #cv2.waitKey(0)
        warped=cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)      #converts rgb to hsv for better color isolation
        mask=cv2.inRange(warped,lowColor,highColor )    #makes a mask for the selected color domains
        warped=cv2.bitwise_and(clean,clean,mask=mask)   #overlays the mask on the clean image to isolate the regions of interest
        warpedCut=cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  #makes a second warp which will remove small contours that may interfere later
        warpedCut=cv2.GaussianBlur(warpedCut,(5,5),2)   #blurs the image to make the countors more appropiate
        warpedCut=cv2.threshold(warpedCut,70,255,cv2.THRESH_BINARY)[1]  #makes a threshold image of the cutting section
        warped=cv2.GaussianBlur(warped,(5,5),5)     #Blurs the color isolated warp to make the image more easily workable
        #cv2.imshow('Gauge Picture1', warpedCut)
        #cv2.waitKey(0)
        warped=cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  #convert to gray
        warped=cv2.threshold(warped,0,255,cv2.THRESH_BINARY)[1] #black and white threshold to get contours from with minimal noise
    

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))   #creates a structure for morphing the image, see CV2 documentation for more info
    
        cnts=cv2.findContours(warpedCut.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #find the contours in the warpedCut that will be removed from the final image (really trying to remove the decimal)
        cnts=im.grab_contours(cnts) 
        remove=[]   #empty array of the elements to remove
        mask = np.ones(warpedCut.shape[:2], dtype="uint8") * 255    #makes a blank masking array that will be overlayed later
        for c in cnts:  
            if cv2.contourArea(c)<80: #if the area is sufficently small then we have an important item
                remove.append(c)    #add the contour to the array of elements to be removed
                cv2.drawContours(mask,[c],-1,0,6)   #draws the element into the mask
        clean2=clean.copy() 
        #cv2.drawContours(clean2, remove, -1, (255,255,0), 8)
        #cv2.imshow('Gauge ', clean2)
        #cv2.imshow('warped1 ', warpedCut)
        warped = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)   #morphs the warped image to remove of some of the threshold noise and make the countours more defined
        warped=cv2.bitwise_and(warped,warped,mask=mask) #removes the small regions from warped cut that can cause issue in the later contours
        #cv2.imshow('Gauge Picture2', warped)
        #cv2.waitKey(0)

        cnts=cv2.findContours(warped.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #finds the contours on the warped copy that will be used for digit recognition
        cnts=im.grab_contours(cnts) 
        cv2.drawContours(warped, cnts, -1, (0,255,0), 1)    #draws the contours onto warped for final showing
        #cv2.imshow('Gauge Picture', warped)
        #cv2.waitKey(0)
        temp=[]     #array for temperature data
        for c in cnts:
            if cv2.contourArea(c)>800: #if the area is sufficently large then we have an important item
                temp.append(c)  #appends large area to the temperature array
        cv2.drawContours(clean, temp, -1, (255,255,0), 1)   #draws the contours on to the clean copy
        #cv2.imshow('Gauge Picture3', clean)
        #cv2.waitKey(0)
        digit=contours.sort_contours(temp, method="left-to-right")[0]   #adds the contours into a tuple which will be the digits
        digits=[]   #blank array for the final digits

        [x,y,w,h]=cv2.boundingRect(digit[0])    #sets the inital conditions for the bounding rectangles

        w_max=55    #maximum width of the digits based on a ratio of the image and digit size from sqFinder

        #print(digit)
        #print([x,y,w,h], len(digit))
        #print(type(digit)) 
        if len(digit)<4:            #if the number of contours in digit is less than the 4 shown on the gauge then it is maually adjusted to 4 with the assumption that the last digit with be most similar in size to the 3rd
            digit=digit+(digit[2],)         

        origin=x    #sets the origin to the base x from contour 0
        #for d in digit:
        i=0     #indexer
    
        for d in digit:
            [a,y,c,h]=cv2.boundingRect(d)
            x=origin+w_max*i        #shifts the x value by i maxmimum digit widths.
            #cv2.rectangle(clean,(x,y),(x+w_max,y+h),(0,50*i,0),2)
            #cv2.rectangle(clean,(x+w-w_max,y),(x+w,y+h),(0,255,0),2)
            #print(d)
            #roi = warped[y:y+h,x:x+w_max]
            roi = warped[y:y+h,x+w-w_max:x+w]#gets the rectangle around the digits
            #now to break the rectangle into smaller sections for comparing to the digits index
            #cv2.drawContours(clean,rect,-1,0,1)
            #print(roi[0])
            x=x+w-w_max #adjust the x value based on the width value
            dW=int(.125*roi.shape[0])   #a box that with exam the digit section width
            #print(dW)
            dH=int(0.15*roi.shape[1]) #a box that with exam the digit section height
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
        
            for (e,((x1,y1),(x2,y2))) in enumerate(breakout): #for each section of the breakout it evaluates if that section has enought light to be considered on
                    #print(e)
                    #print(((x1,y1),(x2,y2)))
                    cv2.rectangle(clean,(x1,y1),(x2,y2),(50*i,50*i,0),5-i)  #draws a rectangle around the section being final image to show the sections being analyzed  
                    sec=warped[y1:y2,x1:x2] #zooms on the section from the breakout on the warped image
                    t=cv2.countNonZero(sec) #counts the non-zeros (colored/white) pixels in the box
                    #print(t)
                    A=(x2-x1)*(y2-y1)   #determine that sections area
                    if(t/A)>.45:    #if the amount of area on is greater than x% of the total area that section is determined on
                        light[e]=1  #TRUE
            digit=DIGITS_LOOKUP[tuple(light)]   #refers to the lookup table to convert to a digit
            digits.append(digit)   
            #print(light)
            #    sum=cv2.countNonZero(sec)

        #print(digits)
        output=""
        for d in digits:
            output=output+str(d)    #makes a sting of the output temperature
        output=output[:-2]+"."+output[-2:]+'\N{DEGREE SIGN}'+'C'    #prints the temp in nice formatting
        print(output)
    except:         #if the image cannot be tabulated then a better image should be provided so it excepts the error and lets the operator know
        print("Please use a more clear image.")
    cv2.imshow('Gauge ref', warped)     #prints the black and white image so the operator can see the image being worked with and maybe find errors
    cv2.imshow('Gauge Contour', clean)  #shows the image with contour boxes so the operator can see what is being analyized and if there are any errors they can fix
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def sqFinder(Img):
    gray=cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)  #convert to blur
    blur=cv2.GaussianBlur(gray,(5,5),0)         #blur the image
    thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1] #if pixel does not meet threshold sets the value to zero
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    #warped = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    edged=cv2.Canny(blur, 50, 200, 255)
    #this image provides enough contrast between the thermometer and backgroud that we can determine it from the other parts of the image
    
    cnts= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=im.grab_contours(cnts)     #makes an array of all the contours from the edged image
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) #sorts the array of contours

    #cv2.imshow('Gauge Picture3', edged)
    #cv2.waitKey(0)

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

    #cv2.imshow('Gauge Picture23', warped)
    #cv2.waitKey(0)

    return val


def circleFxn(circles, b):
    #Finds the center of a circle in cartesian and its radius
    
    avgX=0
    avgY=0
    avgR=0
    for i in range(b):
        avgX = avgX + circles[0][i][0]
        avgY = avgY + circles[0][i][1]
        avgR = avgR + circles[0][i][2]
    avgX = int(avgX/(b))
    avgY = int(avgY/(b))
    avgR = int(avgR/(b))
    return avgX, avgY, avgR

def ptDistance(x1, y1, x2, y2):
    #Finds the distance between two points
    
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist
    
def circleDetect(gaugeImg):
    #Detects the circle of the gauge in the image
    
    height, width = gaugeImg.shape[:2]
    grayImg = cv2.cvtColor(gaugeImg, cv2.COLOR_BGR2GRAY)
    avgCircle = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    a, b, c = avgCircle.shape
    x, y, r = circleFxn(avgCircle, b)
    return x, y, r
    
def lineDetect(gaugeImg, x, y, r):
    #Determines the endpoints of a line lying along the needle
    
    needleLines = []
    
    #Converting image to greyscale and then to binary black and white
    grayImg = cv2.cvtColor(gaugeImg, cv2.COLOR_BGR2GRAY)
    th, binaryImg = cv2.threshold(grayImg, 50, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(image = binaryImg, rho = 10, theta = np.pi / 180, threshold = 100, minLineLength = 10, maxLineGap = 0)
    
    #Parameterizing how close and far the near and far endpoint of the needle
    #line should be to the center of the gauge. Values are scalars multiplied
    #by the radius of the circle which defines the edge of the gauge
    lowerDistLowerBound = 0.01
    lowerDistUpperBound = 0.25
    upperDistLowerBound = 0.5
    upperDistUpperBound = 1.0
    
    for i in range(0, len(lines)):
        
        for x1, y1, x2, y2 in lines[i]:
            dist1 = ptDistance(x, y, x1, y1)  # x, y is center of circle
            dist2 = ptDistance(x, y, x2, y2)  # x, y is center of circle
            
    #Determines which endpoint lies farther from the center of the gauge and
    #makes it the outer endpoint for determining if the near or far points are
    #in the proper range
            
            if (dist1 > dist2):
                temp = dist1
                dist1 = dist2
                dist2 = temp
            
            
            #If the line is in the preset range, add it to a list of valid
            #lines
            if (((dist1<lowerDistUpperBound*r) and (dist1>lowerDistLowerBound*r) and (dist2<upperDistUpperBound*r)) and (dist2>upperDistLowerBound*r)):
                needleLines.append([x1, y1, x2, y2])

    #Returns the first set of endpoints in the list (is usually the best in
    #experience)
    x1 = needleLines[0][0]
    y1 = needleLines[0][1]
    x2 = needleLines[0][2]
    y2 = needleLines[0][3]
    
    return x1, y1, x2, y2

def lineToValue(centerX, centerY, x1, y1, x2, y2, theta1, theta2, psi1, psi2):
    #Determines the value of the gauge from the line endpoints, gauge center
    #coordinates, and the angles and values of the min and max psi
       
    #Determines which endpoint lies farther from the center of the gauge and
    #makes it the outer endpoint for angle determination
    length1 = ptDistance(centerX, centerY, x1, y1)
    length2 = ptDistance(centerX, centerY, x2, y2)
    if (length1 > length2):
        deltaThetaNeedle = abs((180/3.14159)*(math.atan((y1-centerY)/(x1-centerX))))
        x = x1
        y = y1
    else:
        deltaThetaNeedle = abs((180/3.14159)*(math.atan((y2-centerY)/(x2-centerX))))
        x = x2
        y = y2
    
    #Determines which quadrant of the gauge the needle lies in and adds degrees
    #to put the needle in the correct orientation
    if(y<centerY):
        if(x>centerX):
            thetaNeedle = 225-deltaThetaNeedle
        else:
            thetaNeedle = 45+deltaThetaNeedle
    else:
        if(x>centerX):
            thetaNeedle = 225+deltaThetaNeedle
        else:
            thetaNeedle = 45-deltaThetaNeedle 
    
    #Determines the value the needle is pointing to using linear interpolation
    #off of the minimum and maximum value on the gauge
    psiNeedle = thetaNeedle*(psi2-psi1)/(theta2-theta1)
    return psiNeedle
    

def resizeImg(img, percent):
    #Resizes an image for display on a standard monitor (image displyay for
    #testing purposes)
    
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resizedImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resizedImg

def trial(address):
    #A function for testing purposes that verifies which gauge image is being
    #used, shows an image with the gauge and edge circle and needle superimposed,
    #and prints the psi
    
    gaugeImg = cv2.imread(address)
    originalImg = resizeImg(gaugeImg, 50)
    cv2.imshow('Original image', originalImg)
    cv2.waitKey(0)
    
    x, y, r = circleDetect(gaugeImg)
    cv2.circle(gaugeImg, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(gaugeImg, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)
    circleGaugeImg = resizeImg(gaugeImg, 50)
    cv2.imshow('Imposed circle', circleGaugeImg)
    cv2.waitKey(0)
    
    x1, y1, x2, y2 = lineDetect(gaugeImg, x, y, r)
    cv2.line(gaugeImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    lineGaugeImg = resizeImg(gaugeImg, 50)
    psi = round(lineToValue(x, y, x1, y1, x2, y2, 0, 270, 0, 100),1)
    cv2.putText(lineGaugeImg, str(psi), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    cv2.imshow('Imposed circle, line, and value', lineGaugeImg)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()   
    
    print('PSI is:')
    print(psi)
    print('')

def main(file,type): 
    '''
    provide an image path and the type to analyze;
    1 or "digital"==digital gauge
    0 or "analog"==analog gauge
    '''
    path=os.path.dirname(os.path.os.getcwd())+"\\"+file
    if type=="digital" or type==1:
        digitalReader(cv2.imread(path))
    if type=="analog"or type==0:
        trial(path)
    return 0

main('IMG_3681_5.JPG',"digital")
main('IMG_3678.JPEG',1)
main('IMG_3673.JPEG',"digital")
main('testing.JPEG',1)
main('IMG_3677_2.JPEG',"digital")
main('IMG_3679.JPEG',1)
main('IMG_3675_1.JPEG',"digital")
main('IMG_3672.JPEG',0)
main('IMG_3669.JPEG',"analog")
