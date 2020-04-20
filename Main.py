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
'''

import numpy as np
import cv2
import os
import math
import tkinter
from tkinter import filedialog



IMAGEDIR = './input_images/'



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


def getUserInputDimension():
    #print("Enter your window size:(EX:'5x5')")
    print("Enter a size input: EX: '50x100' to get a 50 by 100 image")
    userinput = input()
    StringDimension = userinput.split("x")

    try:
        if(len(StringDimension)!=2):
            raise Exception("Not a dimension!")
        elif(len(StringDimension)==2and StringDimension[0].isdigit() and StringDimension[1].isdigit()):
            width = int(StringDimension[0])
            height = int(StringDimension[1])
            dimensions = (width,height)
            print(dimensions)
            return dimensions
    except:
        print("Error! Inputted Dimension was not in the form: '<width>x<height>'")

def getIndexesGTEElement(array,element):
    indexes = np.where(array >= element)
    points = list(zip(indexes[0], indexes[1]))
    return points

def getIndexesLTEElement(array,element):
    indexes = np.where(array <= element)
    points = list(zip(indexes[0], indexes[1]))
    return points

def getIndexesForGivenElement(array,element):
    indexes = np.where(array == element)
    points = list(zip(indexes[0], indexes[1]))
    return points

#More efficient SAD using numpy
def SAD2(I,T):
    template_height = T.shape[1]
    template_width = T.shape[0]

    if (isinstance(I[0][0], np.ndarray) or isinstance(I[0][0], list)):
        isColorImage = True
        print("THIS IS A COLOR IMAGE")


    else:
        isColorImage = False
        print("THIS IS A GRAYSCALE IMAGE")
        currentTemplateLayedOverImage =np.lib.stride_tricks.as_strided(I,shape=(I.shape[0]-template_height+1,I.shape[1]-template_width+1,template_height,template_width),strides=I.strides*2)
        foundSADValues = abs(currentTemplateLayedOverImage-T).sum(axis=-1).sum(axis=-1)
        return foundSADValues

    return I

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


def SSD(I,T):
    template_height = T.shape[1]
    template_width = T.shape[0]

    window =np.lib.stride_tricks.as_strided(I,shape=(I.shape[0]-template_height+1,I.shape[1]-template_width+1,template_height,template_width),strides=I.strides*2)
    ssd = np.power(T-window,2).sum(axis=-1).sum(axis=-1)
    #print(ssd)
    return ssd

def SAD(I,T):
    template_height = T.shape[1]
    template_width = T.shape[0]

    window =np.lib.stride_tricks.as_strided(I,shape=(I.shape[0]-template_height+1,I.shape[1]-template_width+1,template_height,template_width),strides=I.strides*2)
    sad = abs(window-T).sum(axis=-1).sum(axis=-1)
    return sad

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

    indexes = np.where(corners2>(0.01*corners2.max()))
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

def matchFeaturesTwoImages(leftImage,rightImage,windowSize:int=10):
    leftImageColored = cv2.cvtColor(leftImage,cv2.COLOR_GRAY2RGB)
    rightImageColored = cv2.cvtColor(rightImage,cv2.COLOR_GRAY2RGB)

    pointsLeftImage = HarrisCorner(leftImage)
    print("Left Points:")
    print(pointsLeftImage)
    pointsRightImage = HarrisCorner(rightImage)
    print("Right Points:")
    print(pointsRightImage)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    print("LEFT IMAGE:")
    print(len(leftImage.shape))

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(leftImageColored, None)
    kp2, des2 = orb.detectAndCompute(rightImageColored, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    #Match descriptors.
    matches = bf.match(des1, des2)

    #Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    leftImageuint8 = cv2.convertScaleAbs(leftImage)
    rightImageuint8 = cv2.convertScaleAbs(rightImage)

    print("MATCHES:")
    img3 = cv2.drawMatches(leftImageuint8, kp1, rightImageuint8, kp2, matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    displayImageGivenArray(img3,windowTitle='MATCHES')

def getSumOfAbsoluteDifferences(image1Array:np.ndarray,image2Array:np.ndarray):
    return

def getNSmallestValues(array,n:int):
    flattentedarray = array.flatten()
    print("SMALLEST:"+str(flattentedarray))

    return np.partition(flattentedarray,n)

def getNLargestValues(array,n:int):
    flattentedarray = array.flatten()
    print(np.partition(flattentedarray,-n)[-n:])
    return np.partition(flattentedarray,-n)[-n:]

def drawRectanglesAtGivenPointsTemplate(I,points,template):
    for point in points:
        drawRectangleOnImageGivenTemplate(I,point,template)
    return I

# def LetUserChooseMatchingScore(I,T):
#     useroption = ""
#
#     selectingOption = True
#
#     while(useroption!='exit'):
#         if(selectingOption):
#             print("Select which matching score technique to use:\n"
#                   "1)SAD\n"
#                   "2)SSD\n"
#                   "3)NCC\n"
#                   "(enter the digit)")
#             useroption = input()
#
#             if(useroption.isdigit()==False):
#                 print("Invalid option! (needs to be a digit: 1,2,3)")
#             else:
#                 if(useroption=="1"):
#                     print("You selected SAD")
#                     amountofvalues = 50
#
#                     foundSADValues = SAD(I,T)
#                     smallestSADvalues = getNSmallestValues(foundSADValues, amountofvalues)
#                     points = getIndexesLTEElement(foundSADValues, smallestSADvalues[amountofvalues-1])
#
#                     displayImageGivenArray(drawRectanglesAtGivenPointsTemplate(I, points, T), windowTitle='SAD')
#
#                 elif(useroption=="2"):
#                     print("You selected SSD")
#                     amountofvalues = 1
#
#                     foundSSDValues = SSD(I,T)
#                     smallestSSDvalues = getNSmallestValues(foundSSDValues, amountofvalues)
#                     points = getIndexesLTEElement(foundSSDValues, smallestSSDvalues[amountofvalues-1])
#
#                     displayImageGivenArray(drawRectanglesAtGivenPointsTemplate(I, points, T), windowTitle='SSD')
#                 elif(useroption=="3"):
#                     print("You selected NCC")
#                     amountofvalues = 1
#
#                     foundNCCValues = NCC(I,T)
#                     biggestNCCvalues = getNLargestValues(foundNCCValues, amountofvalues)
#                     print(biggestNCCvalues)
#
#                     #points = getIndexesLTEElement(foundNCCValues, biggestNCCvalues[amountofvalues-1])
#                     points = getIndexesForGivenElement(foundNCCValues,np.amax(foundNCCValues))
#
#                     print("DISPLAYING")
#                     displayImageGivenArray(drawRectanglesAtGivenPointsTemplate(I, points, T), windowTitle='NCC')
#                 break


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


def disparity(left,right,templateSize:int=3,windowSize:int=5,stepSize:int=1,method='SSD', useCV2Instead=True):
    image_height =left.shape[0]
    image_width =left.shape[1]

    halfTemplateSize = int(templateSize/2)
    disparityImage8 = np.zeros_like(left)
    disparityImage = np.zeros_like(left).astype(np.float32)

    print("Started method:"+method)

    for r in range(0,image_height,stepSize):
        print("Finished y:"+(str(r)))
        for c in range(0,image_width,stepSize):
            #print("CURR:"+str((r,c)))

            #Ensures that the indexes are within the image
            startRow = max(0,r-halfTemplateSize)
            startColumn = max(0,c-halfTemplateSize)
            endRow = min(image_height-1,r+halfTemplateSize)
            endColumn = min(image_width-1,c+halfTemplateSize)

            windowStart = max(0, c - int(windowSize / 2))
            windowEnd = min(image_width, c + int((windowSize) / 2))

            window = right[startRow:endRow + 1, windowStart:windowEnd + 1].astype(np.float32) # gets the entire width*templateHeight row in the right image
            #window = (window - np.mean(window))/(window-)

            #Using the start,end indexes, indexing the window out:
            template = left[startRow:endRow+1,startColumn:endColumn].astype(np.float32)


            #print("SSD WINDOW")
            #print(window)

            if (template.shape[0]!=window.shape[0]):
                raise Exception("TEMPLATE HEIGHT!=WINDOW HEIGHT")

            windowValues = None
            correspondingValues = None

            if(method.lower()=='ssd'):
                #print("COMPARING")

                if(useCV2Instead):
                    windowValues = np.array(cv2SSD(window, template))
                else:
                    windowValues = np.array([SSD1D(window, template, keepSlidingWindowWithinImage=True)])


                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)

                correspondingXinRight = windowStart + minLoc[0]
                # print(minVal)
                # print(minLoc)

            elif(method.lower()=='sad'):

                #print("COMPARING")
                windowValues = np.array([SAD1D(window, template, keepSlidingWindowWithinImage=True)])
                # windowValues = np.array(cv2SSD(window, template))

                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)

                correspondingXinRight = windowStart + minLoc[0]
                #print(minVal)
                #print(minLoc)
            elif(method.lower()=='ncc'):
                if(useCV2Instead):
                    windowValues = np.array(cv2NCC(window,template))
                else:
                    windowValues = np.array([NCC1D(window, template, keepSlidingWindowWithinImage=True)])

                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)
                correspondingXinRight = windowStart+maxLoc[0]
            else:
                print("Did not select a method!(SAD,SSD,NCC)")






            #print("X LEFT:"+str(c))
            #print("X RIGHT:"+str(correspondingXinRight))
            #print("MAX:")

            # print("leftx:"+str(c))
            # print("rightx:"+str(correspondingXinRight))

            #disparityValue = abs(correspondingXinRight-c) #Subtract right image location x by left image location x
            disparityValue = c-correspondingXinRight # Subtract left image x from right image x
            #print("DISPARITY VALUE:" + str(disparityValue))
            disparityImage[r][c]=int(disparityValue)





    # print(disparityImage)
    # print(disparityImage.max())
    # print(disparityImage.min())

    #disparityImage = disparityImage+abs(disparityImage.min()) #shifts disparity values so that they are all 0 minimum
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

    disparitymap = cv2.resize(disparitymap,newSize,interpolation=cv2.INTER_CUBIC)

    #Normalize:

    normalized = disparitymap * (1 / disparitymap.max())

    return normalized



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

    topLevelLeft = leftGaussianPyramid[len(leftGaussianPyramid)-1]
    topLevelRight = rightGaussianPyramid[len(rightGaussianPyramid) - 1]


    disparityPyramid = []
    topLevelDisparity = disparity(topLevelLeft, topLevelRight,templateSize=3,windowSize=12,stepSize=1,method=method)
    displayImageGivenArray(topLevelDisparity,windowTitle='TOP LEVEL DISPARITY')
    np.set_printoptions(threshold=np.inf)
    #print("TOP LEVEL DISPARITY:")
    #print(topLevelDisparity)

    #print("DOUBLED DISPARITY VALUES:")
    currLevelDisparity = topLevelDisparity
    disparityPyramid.append(currLevelDisparity)
    for i in range(levels-1):
        currLevelDisparity = propogateDisparityMap(currLevelDisparity)
        displayImageGivenArray(currLevelDisparity)
        disparityPyramid.append(currLevelDisparity)



    print("DISPARITY PYRAMID LEVELS:"+str(len(disparityPyramid)))
    displayImageGivenArray(disparityPyramid[len(disparityPyramid)-1])






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

leftToRight = disparity(ExampleImage3,ExampleImage32,3,15)
#displayImageGivenArray(leftToRight)
rightToLeft = disparity(ExampleImage32,ExampleImage3,3,15)
#displayImageGivenArray(rightToLeft)

print("LEFT TO RIGHT:")
print(leftToRight)

print("RIGHT TO LEFT:")
print(rightToLeft)
exit(0)


ExampleImage2 = np.array([
                    [2, 7, 5, 8, 6,2, 7, 2, 5, 5],
                    [1, 7, 4, 2, 7,2, 7, 4, 0, 7]
                ],dtype=np.uint8)

#slidingWindow(ExampleImage,3,1,keepSlidingWindowWithinImage=True)

listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR, getabspaths=True)
#print(listOfImages)
stopsignimage = getImageArray(getImageFromListOfImages(listOfImages,'stopsign'),True)
streetimage = getImageArray(getImageFromListOfImages(listOfImages,'street'),True)


#Sets up tkinter root windows and closed it (only need it for user browsing)
root = tkinter.Tk()
root.withdraw()

#print("Browse and pick a template image")
#templatepath = browseImagesDialog(MessageForUser='Select your Template (Foreground)')
# #template = getImageArray(templatepath,True)
# template = stopsignimage
# templateOriginalSize = (template.shape[1],template.shape[0])
# print("Displaying your template image, size:"+str(templateOriginalSize))
#
# displayImageGivenArray(template, windowTitle='Template:',waitKey=0)


#Method1: Sliding Template T(x,y) across an Image I(x,y)
#We then want to get 'matches' for the template on the image itself
#print("Enter a desired size for your template image:")
#templateSize = getUserInputDimension()
#templateSize = (10,10)
#template = ScaleByGivenDimensions(template,templateSize)
#displayImageGivenArray(template,windowTitle='Template(resized):',waitKey=0)

#print("\n\n")

#print("Pick your matching window image:")
#imagepath = browseImagesDialog(MessageForUser='Select your Template (Foreground)')
#image = getImageArray(imagepath,True)
image = streetimage
# print("IMAGE:")
# print(image)
# imageoriginalsize = (image.shape[1],image.shape[0])
# print("Displaying your matching window image, size:"+str(imageoriginalsize))
#displayImageGivenArray(image,waitKey=0)


#Performs harris corner on the image to detect corners
harriscornerimage = HarrisCorner(image,True)

#Lets the user pick which matching score they want to use:
#LetUserChooseMatchingScore(image,template)

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

#exit(0)


#displayImageGivenArray(leftImage,windowTitle='Left Image',waitKey=0)
#displayImageGivenArray(rightImage,windowTitle='Right Image',waitKey=0)

print("--------")
print(NCC1D(ExampleImage,ExampleTemplate,keepSlidingWindowWithinImage=True))
print(cv2NCC(ExampleImage,ExampleTemplate))
#exit(0)

print(SSD1D(ExampleImage,ExampleTemplate,keepSlidingWindowWithinImage=True))
#SSD1D(ExampleImage2,ExampleTemplate2,keepSlidingWindowWithinImage=True)
#NCC1D(ExampleImage,ExampleTemplate3,keepSlidingWindowWithinImage=True)

#newSAD(leftImage,ExampleTemplate)
displayImageGivenArray(leftImage)
displayImageGivenArray(rightImage)

harriscornerimage = HarrisCorner(leftImage,True)
harriscornerimage = HarrisCorner(rightImage,True)

matchFeaturesTwoImages(leftImage,rightImage)
#disparityPyramid(leftImage,rightImage,4,method='ncc')
#displayImageGivenArray(disparity2(leftImage,rightImage,templateSize=3,windowSize=25,stepSize=1),windowTitle='Disparity Image',waitKey=0)
#displayImageGivenArray(disparity2(leftImage,rightImage,templateSize=3,windowSize=50,stepSize=1),windowTitle='Disparity Image',waitKey=0)
#displayImageGivenArray(disparity2(leftImage,rightImage,templateSize=11,windowSize=100,stepSize=1),windowTitle='Disparity Image',waitKey=0)
#displayImageGivenArray(disparity2(leftImage,rightImage,templateSize=15,windowSize=200,stepSize=1),windowTitle='Disparity Image',waitKey=0)

#Works for SAD:
#displayImageGivenArray(disparity(leftImage,rightImage,templateSize=3,windowSize=30,stepSize=1,method='sad',useCV2Instead=True),windowTitle='SAD Disparity Image')

#Works for SSD:
#displayImageGivenArray(disparity(leftImage,rightImage,templateSize=3,windowSize=100,stepSize=1,method='ssd',useCV2Instead=True),windowTitle='SSD Disparity Image')
#displayImageGivenArray(disparity(leftImage,rightImage,templateSize=9,windowSize=100,stepSize=1,method='ssd',useCV2Instead=True),windowTitle='SSD Disparity Image') #Looks good
displayImageGivenArray(disparity(leftImage,rightImage,templateSize=15,windowSize=300,stepSize=1,method='ssd',useCV2Instead=True),windowTitle='SSD Disparity Image')

#Works for NCC: (Takes a while to finish running so I use CV2 instead)
#displayImageGivenArray(disparity(leftImage,rightImage,templateSize=11,windowSize=100,stepSize=1,method='ncc',useCV2Instead=True),windowTitle='NCC Disparity Image',waitKey=0)