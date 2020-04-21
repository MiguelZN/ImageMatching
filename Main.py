'''
Miguel Zavala
4/15/20
CISC442 Computer Vision
PR2

Design and implement Matlab or C or Python program for a stereo analysis system involving
feature-based, region-based and multiresolution matching. The program should be able to
perform multi-resolution stereo analysis, where the no. of levels are set by the user.
The template and search neighborhood at each level can be set differently by the user,
so is the method and matching measure to be used

Note: I used the actual CV2 SSD, NCC methods since they are far more efficent but I did create my own as well and produce the same values
'''

import numpy as np
import cv2
import os
import math
import tkinter
from scipy import ndimage

from tkinter import filedialog

a = np.array([[13, 21, 13,  8],
              [ 5, 10, 22, 14],
              [21, 33,  9,  0],
              [ 0,  0,  0,  0]], dtype=np.float)

IMAGEDIR = './input_images/'



def neighborhoodAverage(array,kernelSize:int=3):
    result = ndimage.generic_filter(array, np.nanmean, size=kernelSize, mode='constant', cval=np.NaN)
    print(result)
    return result



def neighborhoodAverage2(array,point,kernelSize:int=3):
    halfKernelSize = int((kernelSize-1)/2)
    centerpoint = point
    x = centerpoint[0]
    y = centerpoint[1]

    array_width = array.shape[1]
    array_height = array.shape[0]

    print(array.shape)

    startRow = max(0, y - halfKernelSize)
    startColumn = max(0, x - halfKernelSize)
    endRow = min(array_height - 1, y + halfKernelSize)
    endColumn = min(array_width - 1, x + halfKernelSize)

    window = array[startRow:endRow+1,startColumn:endColumn+1]
    array[y][x] = window.mean()

# neighborhoodAverage2(a,(0,0),3)
# print(a)


#Searches a list of strings and looks for specified image name (string)
def getImageFromListOfImages(listofimages, want):
    for imagepath in listofimages:
        print(imagepath)
        if(want in imagepath):
            return imagepath

    raise Exception('Could not find image with this name!')

#Takes a path directory (string) and checks for all images in that directory
#Returns a list of image paths (list of strings)
def getAllImagesFromInputImagesDir(path:str, getabspaths=True):
    listOfImagePaths = []

    if (path[0] == '.' and getabspaths):
        path = os.getcwd() + path[1:path.__len__()]


    # read the entries
    with os.scandir(path) as listOfEntries:
        curr_path = ""
        for entry in listOfEntries:
            # print all entries that are files
            if entry.is_file() and ('png' in entry.name.lower() or 'jpg' in entry.name.lower()):
                #print(entry.name)

                if (getabspaths):
                    curr_path=path+entry.name
                    #print(path)
                else:
                    curr_path = entry.name


                listOfImagePaths.append(curr_path)

    return listOfImagePaths

#Takes in an imagepath (string) and displays the image
def displayImageGivenPath(imagepath:str, wantGrayImage=False):
    img = getImageArray(imagepath,wantGrayImage)
    cv2.imshow('image', img)
    cv2.waitKey(0) #waits for user to type in a key
    cv2.destroyAllWindows()

#Takes in an image (np array) and displays the image
def displayImageGivenArray(numpyimagearr, windowTitle:str='image', waitKey:int=0):
    cv2.imshow(windowTitle, numpyimagearr)

    if(waitKey==0):
        print("(Press any key to continue...)")
    else:
        print("Displaying image for "+str(waitKey/1000)+"seconds")
    cv2.waitKey(waitKey)  # waits for user to type in a key
    cv2.destroyAllWindows()

#Takes in an image path (string) and returns the image as a np array
def getImageArray(imagepath:str, intensitiesOnly=True):
    if(intensitiesOnly):
        return cv2.imread(imagepath, 0)
    else:
        return cv2.imread(imagepath)

#Takes in two images (np arrays) and scales image 1 to the exact dimensions of image 2
#(did this because sometimes dividing by 2 would result in different sized matrices due to
#rounding or ceiling or flooring by python)
def ScaleImage1ToImage2(image1,image2):
    # print(image1)
    # print(image2)

    newimage1 = None
    if(image1.shape[0]!=image2.shape[0] or image1.shape[1]!=image2.shape[1]):
        newimage1 = cv2.resize(image1, (image2.shape[1],image2.shape[0]))
        return newimage1
    else:
        return image1

def ScaleByGivenDimensions(I,dim):
    new_image = cv2.resize(I, (dim[0],dim[1]))
    return new_image

#Allows the user to manually select images
def browseImagesDialog(startindirectory:str=os.getcwd(),MessageForUser='Please select a file'):
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    filepath = filedialog.askopenfilename(parent=root, initialdir=startindirectory, title=MessageForUser,filetypes = (("png files","*.png"),("jpeg files","*.jpg"),("all files","*.*")))
    return filepath


def getUserInputDimension(message = "Enter a template size: EX: '3'"):
    #print("Enter your window size:(EX:'5x5')")
    print(message)
    userinput = input()
    if(userinput.isdigit()):
        print('Entered size:'+userinput)
        return int(userinput)

def drawRectangleOnImage(I,topLeftPoint,dimension):
    # print(I)
    # print(topLeftPoint)
    # print(dimension)

    I = cv2.rectangle(I,topLeftPoint,dimension,(255,0,0),2)
    return I

def drawRectangleOnImageGivenTemplate(I,topLeftPoint,template):
    I=drawRectangleOnImage(I,topLeftPoint,(topLeftPoint[0]+template.shape[1],topLeftPoint[1]+template.shape[0]))
    return I

def cv2NCC(I,T):
    ncc = cv2.matchTemplate(I, T, cv2.TM_CCORR_NORMED)
    return ncc

def cv2SSD(I,T):
    ssd = cv2.matchTemplate(I, T, cv2.TM_SQDIFF)
    return ssd


#threshold is between [0,255], to get good points use ~200
def HarrisCorner2(I, threshold:int=200, displayImage:bool=False):
    imagecopy = np.copy(I)
    cornerpoints=cv2.cornerHarris(imagecopy, 2, 3, 0.055)
    #cornerpoints = cv2.cornerHarris(imagecopy, 7, 5, 0.05)
    #cornerpoints = cv2.cornerHarris(imagecopy, 9, 7, 0.055)


    imagecopy = np.empty(cornerpoints.shape,dtype=np.float32)
    cv2.normalize(cornerpoints, imagecopy, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    imagecopy = cv2.convertScaleAbs(imagecopy)

    #draws circles around points detected by harris
    for i in range(imagecopy.shape[0]):
        for k in range(imagecopy.shape[1]):
            if int(imagecopy[i,k])>threshold:
                point = (k,i)
                thickness = 2
                radius = 5
                color = (0)
                cv2.circle(imagecopy, point,radius,color,thickness)

    if(displayImage):
        displayImageGivenArray(imagecopy)

    return imagecopy

def HarrisCorner(I,displayImage:bool=False):
    imagecopy = np.copy(I)
    corners = cv2.cornerHarris(I, 2, 3, 0.04)

    corners2 = cv2.dilate(corners, None, iterations=3)
    print(corners2)

    indexes = np.where(corners2>(0.005*corners2.mean()))
    points = list(zip(indexes[0], indexes[1]))
    print(indexes)

    imagecopy = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)
    print("IMG SIZE:")
    print(imagecopy.shape)
    displayImageGivenArray(imagecopy,windowTitle='GRAY TO COLOR')

    for point in points:
        imagecopy[point[0]][point[1]] = [0,0,255]


    if(displayImage):
        displayImageGivenArray(imagecopy)

    return points


#Uses Harris Corner to find points in both the left and right images individually
#Loops through all left image points and uses region SSD,SAD,or NCC to find corresponding point
#(Note: takes a long time to run)
def featureBasedMethod(leftImage,rightImage,templateSize:int=3,windowFeatureSize:int=4,windowRegionSize:int=50,method='ssd'):
    pointsLeftImage = HarrisCorner(leftImage)
    print("Left Points:")
    print(pointsLeftImage)
    pointsRightImage = HarrisCorner(rightImage)
    print("Right Points:")
    print(pointsRightImage)

    disparityImage = np.zeros_like(leftImage,np.float32)

    #Matching left points to right points and then checking for similarity using
    currentRow = pointsLeftImage[0][0]
    image_width = leftImage.shape[1]
    image_height = leftImage.shape[0]

    halfTemplateSize = int((templateSize-1)/2)

    startRightPoints=0
    for leftpoint in pointsLeftImage:
        centerpoint = leftpoint
        leftx = centerpoint[1]
        lefty = centerpoint[0]

        if (lefty != currentRow):
            currentRow = lefty+1

        startRow = max(0, lefty - halfTemplateSize)
        startColumn = max(0, leftx - halfTemplateSize)
        endRow = min(image_height - 1, lefty + halfTemplateSize)
        endColumn = min(image_width - 1, leftx + halfTemplateSize)

        #Constructs a template region area around the point in the left image
        template = leftImage[startRow:endRow+1,startColumn:endColumn+1]

        for i in range(startRightPoints,len(pointsRightImage)):
            rightpoint = pointsRightImage[len(pointsRightImage)-1]
            rightx = rightpoint[1]
            righty = rightpoint[0]


            if(righty!=lefty):
                break

            startWindowRow = lefty
            halfWindowSize = int((windowFeatureSize-1)/2)
            startWindowColumn = max(0, rightx-halfWindowSize)
            endWindowRow = min(image_height - 1, lefty + halfTemplateSize)
            endWindowColumn = min(image_width - 1, rightx + halfWindowSize)

            window = rightImage[startWindowRow:endWindowRow+1,startWindowColumn:endWindowColumn+1]

            correspondingValues = None
            if(method.lower()=='ssd'):
                correspondingValues = cv2SSD(window,template)

            elif(method.lower()=='ncc'):
                correspondingValues = cv2SSD(window, template)

            else:
                correspondingValues = SAD1D(window,template)

            print("CORRESPONDING VALUES:"+str(correspondingValues))
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(correspondingValues)

            correspondingRightX = startWindowColumn+minLoc[0]
            print("RIGHT X:"+str(correspondingRightX))

            disparityValue = leftx-correspondingRightX
            print(disparityValue)
            disparityImage[lefty][leftx]=disparityValue

    #Once finished going through all feature points, uses normal region based disparity to fill the rest
    disparityImage= disparity(leftImage, rightImage,
                  templateSize= templateSize, windowSize=windowRegionSize, stepSize= 1, method = method, useCV2Instead = True, validityCheck = False, existingDisparityImage = disparityImage)

    displayImageGivenArray(disparityImage,windowTitle='Finished Feautre Based Disparity')
    return disparityImage


# def drawRectanglesAtGivenPointsTemplate(I,points,template):
#     for point in points:
#         drawRectangleOnImageGivenTemplate(I,point,template)
#
#     return I



#Lets the user choose images, template size, window size, and which methods to use
def LetUserChoose():
    valueTechnique = ""
    method=""
    leftImagePath = ""
    rightImagePath = ""
    # leftImage = None
    # rightImage = None


    selectingOption = True

    while(valueTechnique!='exit' or method!='exit'):
        if(selectingOption):
            templateSize = getUserInputDimension('Enter a template size:(1 digit)')
            windowSize = getUserInputDimension('Enter a window size:(1 digit)')

            print("Select method type:\n1)Region Based(Template,Windows)"
                  "\n2)Feature Based(Harris Corner)")

            method = input()

            print("Select which matching score technique to use:\n"
                  "1)SAD\n"
                  "2)SSD\n"
                  "3)NCC\n"
                  "(enter the digit)")
            valueTechnique = input()

            leftImagePath = browseImagesDialog(IMAGEDIR,'Select your left image')
            rightImagePath = browseImagesDialog(IMAGEDIR,'Select your right image')
            # leftImagePath = getImageFromListOfImages(listOfImages, 'scene1.row3.col1')
            # rightImagePath = getImageFromListOfImages(listOfImages, 'scene1.row3.col5')

            # Returns the images as float 32 (but need to normalize to get them to between [0,1]
            leftImage = getImageArray(leftImagePath, intensitiesOnly=True).astype(np.float32)
            rightImage = getImageArray(rightImagePath, intensitiesOnly=True).astype(np.float32)
            leftImage = leftImage / 128
            rightImage = rightImage / 128

            if(valueTechnique.isdigit()==False):
                print("Invalid option! (needs to be a digit: 1,2,3)")

            if (method.isdigit() == False):
                print("Invalid option! (needs to be a digit: 1,2)")
            else:
                if(valueTechnique=="1"):
                    print("You selected SAD")

                    if(method=='2'):
                        displayImageGivenArray(featureBasedMethod(leftImage,rightImage,templateSize,4,windowSize,'sad'),windowTitle='SAD|TemplateSize:'+str(templateSize)+"|WindowSize:"+str(windowSize))
                    else:
                        displayImageGivenArray(disparity(leftImage,rightImage,templateSize,windowSize,1,'sad',True,False,None),windowTitle='SAD|TemplateSize:'+str(templateSize)+"|WindowSize:"+str(windowSize))


                elif(valueTechnique=="2"):
                    print("You selected SSD")

                    if (method == '2'):
                        displayImageGivenArray(featureBasedMethod(leftImage, rightImage, templateSize, 4, windowSize, 'ssd'),windowTitle='SSD|TemplateSize:'+str(templateSize)+"|WindowSize:"+str(windowSize))
                    else:
                        displayImageGivenArray(disparity(leftImage, rightImage, templateSize, windowSize, 1, 'ssd', True, False, None),windowTitle='SSD|TemplateSize:'+str(templateSize)+"|WindowSize:"+str(windowSize))

                elif(valueTechnique=="3"):
                    print("You selected NCC")

                    if (method == '2'):
                        displayImageGivenArray(featureBasedMethod(leftImage, rightImage, templateSize, 4, windowSize, 'ncc'),windowTitle='NCC|TemplateSize:'+str(templateSize)+"|WindowSize:"+str(windowSize))
                    else:
                        displayImageGivenArray(disparity(leftImage, rightImage, templateSize, windowSize, 1, 'ncc', True, False, None),windowTitle='NCC|TemplateSize:'+str(templateSize)+"|WindowSize:"+str(windowSize))

                break


def slidingWindow(I,windowSize:int=3,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]

    halfWindowSize = int(windowSize/2)
    #print(halfWindowSize)

    AllWindows = []

    if(keepSlidingWindowWithinImage==False):
        for r in range(0,image_height,stepSize):
            for c in range(0,image_width,stepSize):
                print("CURR:"+str((r,c)))

                #Ensures that the indexes are within the image
                startRow = max(0,r-halfWindowSize)
                startColumn = max(0,c-halfWindowSize)
                endRow = min(image_height,r+halfWindowSize)
                endColumn = min(image_width,c+halfWindowSize)

                print("Start:"+str((startRow,startColumn)))
                print("End:"+str((endRow,endColumn)))

                #Using the start,end indexes, indexing the window out:
                window = I[startRow:endRow+1,startColumn:endColumn+1]
                print(window)
                print('---')
                AllWindows.append(window)
    else:
        for r in range(0, image_height, stepSize):
            for c in range(0, image_width, stepSize):
                print("CURR:" + str((r, c)))


                # Ensures that the indexes are within the image
                startRow = r
                startColumn = c
                endRow = r+windowSize
                endColumn = c+windowSize

                if(endRow<=image_height and endColumn<=image_width):

                    print("Start:" + str((startRow, startColumn)))
                    print("End:" + str((endRow, endColumn)))

                    # Using the start,end indexes, indexing the window out:
                    window = I[startRow:endRow, startColumn:endColumn]
                    print(window)
                    print('---')
                    AllWindows.append(window)


    print("All Windows:")
    print(AllWindows)


#SSD calculation method for region based
def SSD1D(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]
    templateHeightSize = T.shape[0]#height
    templateWidthSize = T.shape[1] #width
    if (image_height != T.shape[0]):
        print(image_height)
        print(I)
        print(T)
        raise Exception("WINDOW AND TEMPLATE DO NOT HAVE THE SAME HEIGHT")

    AllWindows = []

    correspondenceValues = []

    for c in range(0, image_width, stepSize):
        # Ensures that the indexes are within the image
        startRow = 0
        startColumn = c
        endRow = templateHeightSize
        endColumn = c+templateWidthSize

        if(endRow<=image_height and endColumn<=image_width):
            # Using the start,end indexes, indexing the window out:
            window = I[startRow:endRow, startColumn:endColumn]
            AllWindows.append(window)
            correspondence = np.sum(np.power(T-window,2))
            correspondenceValues.append(correspondence)
    return correspondenceValues

#SAD calculation method for region based
def SAD1D(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]

    templateHeightSize = T.shape[0]#height
    templateWidthSize = T.shape[1] #width

    AllWindows = []

    correspondenceValues = []

    for c in range(0, image_width, stepSize):
        # Ensures that the indexes are within the image
        startRow = 0
        startColumn = c
        endRow = templateHeightSize
        endColumn = c+templateWidthSize

        if(endRow<=image_height and endColumn<=image_width):
            window = I[startRow:endRow, startColumn:endColumn]
            AllWindows.append(window)
            correspondence = np.sum(np.abs(T-window))
            correspondenceValues.append(correspondence)

    return correspondenceValues

#NCC1D calculation method for region based
def NCC1D(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]

    templateHeightSize = T.shape[0]#height
    templateWidthSize = T.shape[1] #width

    AllWindows = []

    correspondenceValues = []

    for c in range(0, image_width, stepSize):
        # Ensures that the indexes are within the image
        startRow = 0
        startColumn = c
        endRow = templateHeightSize
        endColumn = c+templateWidthSize

        if(endRow<=image_height and endColumn<=image_width):
            window = I[startRow:endRow, startColumn:endColumn]
            AllWindows.append(window)

            #numerator = np.sum(T*window)
            # numerator = (np.sum(np.dot((T-T.mean()),(window-window.mean()))))/(T.size)
            # denominator = np.dot(T.std(),window.std())
            #denominator = np.sqrt(np.sum(np.power(T-T.mean(),2))/T.size)*np.sqrt(np.sum(np.power(window-window.mean(),2))/window.size)

            numerator = (T - T.mean())*(window - window.mean()) #subtracts the mean from each window and multiplies them together
            denominator = T.std()*window.std() #Product of both standard deviations

            correspondence = (np.sum(numerator/denominator))/(T.size) #Divides the numerator by denominator and takes the summation and divdes the summation by n pixels
            correspondenceValues.append(correspondence)

    return correspondenceValues

#Region Based method:
#Uses template and window matching to find corresponding points
#Creates a disparity map image showcasing depth of left and right images
def disparity(left,right,templateSize:int=3,windowSize:int=5,stepSize:int=1,method='SSD', useCV2Instead=True,validityCheck=False,existingDisparityImage=None):
    image_height =left.shape[0]
    image_width =left.shape[1]

    halfTemplateSize = int((templateSize-1)/2)

    disparityImage = None

    if(existingDisparityImage!=None):
        disparityImage = existingDisparityImage
    else:
        # stores all of the disparities and places them in the left image i,j locations
        disparityImage = np.zeros_like(left).astype(np.float32)

    #Keeps all of the disparities and should place disparity values in the left image i,j locations
    validityDisparityImage =np.zeros_like(left).astype(np.float32)

    print("Started method:"+method)


    for r in range(0,image_height,stepSize):
        print("Finished y:"+(str(r)))
        for c in range(0,image_width,stepSize):
            #Continues if an existing disparity map image was given
            if(disparityImage[r][c]!=0):
                continue

            #print("CURR:"+str((r,c)))

            #Ensures that the indexes are within the image
            startRow = max(0,r-halfTemplateSize)
            startColumn = max(0,c-halfTemplateSize)
            endRow = min(image_height-1,r+halfTemplateSize)
            endColumn = min(image_width-1,c+halfTemplateSize)

            windowStart = max(0, c - int((windowSize-1) / 2))
            windowEnd = min(image_width, c + int((windowSize-1) / 2))

            window = right[startRow:endRow + 1, windowStart:windowEnd + 1].astype(np.float32) # gets the entire width*templateHeight row in the right image
            #Using the start,end indexes, indexing the window out:
            template = left[startRow:endRow+1,startColumn:endColumn+1].astype(np.float32)

            validityTemplate = None
            validityWindow = None
            if(validityCheck):
                validityTemplate = right[startRow:endRow+1,startColumn:endColumn+1].astype(np.float32)
                validityWindow = left[startRow:endRow + 1, windowStart:windowEnd + 1].astype(np.float32)

            if (template.shape[0]!=window.shape[0]):
                raise Exception("TEMPLATE HEIGHT!=WINDOW HEIGHT")

            windowValues = None
            correspondingValues = None

            validityWindowValues = None
            validityCorrespondingValues = None
            ValiditycorrespondingXinRight = -1

            if(method.lower()=='ssd'):
                #print("COMPARING")

                if(useCV2Instead):
                    windowValues = np.array(cv2SSD(window, template))

                    if(validityCheck):
                        validityWindowValues = np.array(cv2SSD(validityWindow,validityTemplate))
                        ValidityminVal, ValiditymaxVal, ValidityminLoc, ValiditymaxLoc = cv2.minMaxLoc(validityWindowValues)
                        ValiditycorrespondingXinRight = np.arange(correspondingValues.shape[1])[ValidityminLoc[0]]
                else:
                    windowValues = np.array([SSD1D(window, template, keepSlidingWindowWithinImage=True)])

                    if (validityCheck):
                        validityWindowValues = np.array([SSD1D(validityWindow, validityTemplate, keepSlidingWindowWithinImage=True)])
                        ValidityminVal, ValiditymaxVal, ValidityminLoc, ValiditymaxLoc =cv2.minMaxLoc(validityWindowValues)
                        ValiditycorrespondingXinRight = np.arange(correspondingValues.shape[1])[ValidityminLoc[0]]




                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)

                correspondingXinRight = windowStart + minLoc[0]
                # print(minVal)
                # print(minLoc)

            elif(method.lower()=='sad'):

                #print("COMPARING")
                windowValues = np.array([SAD1D(window, template, keepSlidingWindowWithinImage=True)])



                if(validityCheck):
                    validityWindowValues = np.array([SAD1D(validityWindow, validityTemplate, keepSlidingWindowWithinImage=True)])
                    ValidityminVal, ValiditymaxVal, ValidityminLoc, ValiditymaxLoc = cv2.minMaxLoc(validityWindowValues)
                    ValiditycorrespondingXinRight =np.arange(validityWindowValues.shape[1])[ValidityminLoc[0]]

                # windowValues = np.array(cv2SSD(window, template))

                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)

                correspondingXinRight = windowStart + minLoc[0]
                #print(minVal)
                #print(minLoc)
            elif(method.lower()=='ncc'):
                if(useCV2Instead):
                    windowValues = np.array(cv2NCC(window,template))

                    if (validityCheck):
                        validityWindowValues = np.array(cv2NCC(validityWindow, validityTemplate))
                        ValidityminVal, ValiditymaxVal, ValidityminLoc, ValiditymaxLoc = cv2.minMaxLoc(validityWindowValues)
                        ValiditycorrespondingXinRight = np.arange(validityWindowValues.shape[1])[ValidityminLoc[0]]
                else:
                    windowValues = np.array([NCC1D(window, template, keepSlidingWindowWithinImage=True)])

                    if (validityCheck):
                        validityWindowValues = np.array([NCC1D(validityWindow, validityTemplate, keepSlidingWindowWithinImage=True)])
                        ValidityminVal, ValiditymaxVal, ValidityminLoc, ValiditymaxLoc =cv2.minMaxLoc(validityWindowValues)
                        ValiditycorrespondingXinRight = np.arange(validityWindowValues.shape[1])[ValiditymaxLoc[0]]

                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)
                correspondingXinRight = windowStart + minLoc[0]
            else:
                print("Did not select a method!(SAD,SSD,NCC)")

            disparityValue = c-correspondingXinRight # Subtract left image x from right image x
            validityDisparityValue = c-ValiditycorrespondingXinRight
            #print("DISPARITY VALUE:" + str(disparityValue))
            disparityImage[r][c]=int(disparityValue)
            validityDisparityImage[r][ValiditycorrespondingXinRight]=int(validityDisparityValue) #Places the disparity values on the right image rather than the right image so we can compare values vs left disparity image


    if(validityCheck):
        absoluteDifferenceDisparityImage = np.abs(disparityImage)-np.abs(validityDisparityImage)
        print("ABSOLUTE DIFF DISPARITY IMAGE:")
        print(absoluteDifferenceDisparityImage)

        indexesWhereNotEqual = np.where(absoluteDifferenceDisparityImage!=0)
        unequalPoints = list(zip(indexesWhereNotEqual[0], indexesWhereNotEqual[1]))
        print("POINTS WHERE NOT THE SAME:")
        print(unequalPoints)
        print(len(unequalPoints))
        #exit(0)

        indexesWhereEqual = np.where(absoluteDifferenceDisparityImage == 0)
        equalPoints = list(zip(indexesWhereEqual[0], indexesWhereEqual[1]))
        print("POINTS WHERE THE SAME:")
        print(equalPoints)
        print(len(equalPoints))

        #for point in equalPoints:
        #    disparityImage[point[0]][point[1]]=128

        print(disparityImage[15][10])
        print(validityDisparityImage[15][10])


    #Normalizes values so that they are all within [0,1]
    normalized = disparityImage * (1/disparityImage.max())
    return normalized

'''
Takes in a disparity map and creates the next lower level
by doubling the current disparity values and placing them in the
double sized location
EX: Disparity value at (5,10) = 3, then the next level the
value will be at (10,20) = 6 and the pixels in between will need to 
be fixed with interpolation
'''
def propogateDisparityMap(disparitymap):
    print(disparitymap)
    disparitymap = disparitymap*2
    print("DOUBLED:")
    print(disparitymap)
    disparitymap_height = disparitymap.shape[0]
    disparitymap_width = disparitymap.shape[1]

    newSize = (int(disparitymap_width*2),int(disparitymap_height*2))
    print(newSize)
    print("FDSKFLJSDFK")

    disparitymap = cv2.resize(disparitymap,newSize,interpolation=cv2.INTER_AREA)

    return disparitymap


#Uses gaussian pyramids of images to generate a smoother disparity map image
def disparityPyramid(left, right, levels,method='ssd'):
    leftG = left.copy()
    rightG = right.copy()

    leftGaussianPyramid =[]
    rightGaussianPyramid = []

    #Creates the gaussian pyramids for left,right iamge
    for i in range(levels):
        displayImageGivenArray(leftG)
        leftGaussianPyramid.append(leftG)
        leftG = cv2.pyrDown(leftG)

        rightGaussianPyramid.append(rightG)
        rightG = cv2.pyrDown(rightG)

    print("LEFT LEVELS:"+str(len(leftGaussianPyramid)))

    #Gets the top level gaussian images from both left and right image
    topLevelLeft = leftGaussianPyramid[len(leftGaussianPyramid)-1]
    topLevelRight = rightGaussianPyramid[len(rightGaussianPyramid) - 1]

    displayImageGivenArray(topLevelLeft,windowTitle='TOP LEVEL LEFT')
    displayImageGivenArray(topLevelRight,windowTitle='TOP LEVEL RIGHT')


    #Check validates (gets LeftToRight disparity map, and RightToLeft disparity map and compares)
    #Places 0s where disparity values do not match:
    disparityPyramid = []
    topLevelLeftToRightDisparity = disparity(topLevelLeft, topLevelRight,templateSize=3,windowSize=50,stepSize=1,method=method)
    displayImageGivenArray(topLevelLeftToRightDisparity,windowTitle='Disparity Map:Left to Right')
    #topLevelRightToLeftDisparity = disparity(topLevelLeft, topLevelRight, templateSize=3, windowSize=12, stepSize=1,method=method)
    topLevelDisparity = topLevelLeftToRightDisparity
    #topLevelDisparity = checkDisparityImagesValidity(topLevelLeftToRightDisparity, topLevelRightToLeftDisparity)

    #Neighborhood averaging (Fills in 0s with average neighborhood):
    print("TOP LEVEL SHAPE:"+str(topLevelDisparity.shape))
    #topLevelDisparity = neighborhoodAverage(topLevelDisparity,3) #3 kernel size
    zeroIndexes = np.where((topLevelDisparity <=0.2))
    oneIndexes = np.where((topLevelDisparity>=0.6))
    allIndexes = zeroIndexes+oneIndexes
    allIndexes = list(zip(allIndexes[1], allIndexes[0]))
    print(allIndexes)
    for point in allIndexes:
        x = point[0]
        y = point[1]

        print("VALUE BEFORE:"+str(topLevelDisparity[y][x]))

        neighborhoodAverage2(topLevelDisparity,point,15)

        print("VALUE AFTER:"+str(topLevelDisparity[y][x]))
        print("AVERAGING")





    displayImageGivenArray(topLevelDisparity,windowTitle='Neighborhood averaged:')
    print("NEIGHBORHOOD SHAPE:"+str(topLevelDisparity.shape))

    np.set_printoptions(threshold=np.inf)

    #Goes through each disparity level and propogates disparity map
    currLevelDisparity = topLevelDisparity
    disparityPyramid.append(currLevelDisparity)
    for i in range(levels-1):
        currLevelDisparity = propogateDisparityMap(currLevelDisparity)
        displayImageGivenArray(currLevelDisparity)
        disparityPyramid.append(currLevelDisparity)



    #Normalize:
    fullsizeDisparityImage = disparityPyramid[len(disparityPyramid)-1]
    normalized = fullsizeDisparityImage * (1 / fullsizeDisparityImage.max())


    print("DISPARITY PYRAMID LEVELS:"+str(len(disparityPyramid)))
    displayImageGivenArray(normalized,windowTitle='Finished Disparity Image:',waitKey=10000)

    return normalized


#Method used for checking disparity between methods
def checkDisparityImagesValidity(disparityimage1,disparityimage2):
    validityDisparityImage = np.zeros(disparityimage1.shape,dtype=disparityimage1.dtype)
    for i in np.arange(disparityimage1.shape[0]):
        for j in np.arange(disparityimage1.shape[1]):
            if(disparityimage1[i][j]==disparityimage2[i][j]):
                validityDisparityImage[i][j] = disparityimage1[i][j]
            else:
                validityDisparityImage[i][j] = 0
    return validityDisparityImage


ExampleTemplate = np.array([
                    [2, 5, 5],
                    [4, 0, 7],
                    [7, 5, 9]
                ],dtype=np.uint8)

ExampleTemplate2 = np.array([
                    [2, 5],
                    [4, 0]
                ],dtype=np.uint8)

ExampleTemplate3 = np.array([
                    [2, 5],
                    [4, 0],
                    [7, 5]
                ],dtype=np.uint8)


print("EXAMPLE:")
ExampleImage = np.array([
                    [2, 7, 5, 8, 6],
                    [1, 7, 4, 2, 7],
                    [8, 4, 6, 8, 5]
                ],dtype=np.uint8)

ExampleImage3 = np.array([
                    [2, 7, 5, 8, 6,2, 7, 2, 5, 5],
                    [1, 7, 4, 2, 7,2, 7, 4, 0, 7],
                    [8, 4, 6, 8, 5,2, 7, 7, 5, 9]
                ],dtype=np.uint8)

ExampleImage32 = np.array([
                    [0, 5, 6, 2, 7, 5, 8, 6, 2, 7],
                    [0, 5, 6, 2, 7, 2, 7, 4, 0, 7],
                    [8, 4, 6, 8, 5, 2, 7, 7, 5, 9]
                ],dtype=np.uint8)

#leftToRight = disparity(ExampleImage3,ExampleImage32,3,15)
#displayImageGivenArray(leftToRight)
#rightToLeft = disparity(ExampleImage32,ExampleImage3,3,15)
#displayImageGivenArray(rightToLeft)

print("LEFT TO RIGHT:")
#print(leftToRight)

print("RIGHT TO LEFT:")
#print(rightToLeft)


ExampleImage2 = np.array([
                    [2, 7, 5, 8, 6,2, 7, 2, 5, 5],
                    [1, 7, 4, 2, 7,2, 7, 4, 0, 7]
                ],dtype=np.uint8)

#slidingWindow(ExampleImage,3,1,keepSlidingWindowWithinImage=True)

listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR, getabspaths=True)


#Sets up tkinter root windows and closed it (only need it for user browsing)
root = tkinter.Tk()
root.withdraw()

#Method1: Sliding Template T(x,y) across an Image I(x,y)
#We then want to get 'matches' for the template on the image itself
#print("Enter a desired size for your template image:")
#templateSize = getUserInputDimension()
#templateSize = (10,10)

#leftImagePath = browseImagesDialog(IMAGEDIR,'Select your left image')
#rightImagePath = browseImagesDialog(IMAGEDIR,'Select your right image')
leftImagePath = getImageFromListOfImages(listOfImages,'scene1.row3.col1')
rightImagePath = getImageFromListOfImages(listOfImages,'scene1.row3.col5')

#Returns the images as float 32 (but need to normalize to get them to between [0,1]
leftImage = getImageArray(leftImagePath,intensitiesOnly=True).astype(np.float32)
rightImage = getImageArray(rightImagePath,intensitiesOnly=True).astype(np.float32)
leftImage = leftImage/128
rightImage = rightImage/128
#----------------------





#Allows user to pick their images
LetUserChoose()

#Harris Corner:
harriscornerimage = HarrisCorner(leftImage,True)
harriscornerimage = HarrisCorner(rightImage,True)

#Feature Based Method:(Takes a long time to run so I commented it out)
#featureBasedMethod(leftImage,rightImage,templateSize=3,windowFeatureSize=4)

#Gaussian Pyramid Example:
print("Displaying left inputted image for gaussian:")
displayImageGivenArray(leftImage,waitKey=1000)

print("Displaying right inputted image for gaussian:")
displayImageGivenArray(rightImage,waitKey=1000)
disparityPyramid(leftImage,rightImage,3,method='ncc')
