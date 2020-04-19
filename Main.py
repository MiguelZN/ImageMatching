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
    points = list(zip(indexes[1], indexes[0]))
    return points

def getIndexesLTEElement(array,element):
    indexes = np.where(array <= element)
    points = list(zip(indexes[1], indexes[0]))
    return points

def getIndexesForGivenElement(array,element):
    indexes = np.where(array == element)
    points = list(zip(indexes[1], indexes[0]))
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

def NCC(I,T):
    template_height = T.shape[1]
    template_width = T.shape[0]
    template_area = template_width*template_height
    template_mean = np.mean(T)
    # print("TEMPLATE MEAN VALUE:"+str(template_mean))
    #
    # window = np.lib.stride_tricks.as_strided(I, shape=(
    # I.shape[0] - template_height + 1, I.shape[1] - template_width + 1, template_height, template_width),
    #                                        strides=I.strides * 2)
    # templateSubMean = (T-template_mean)
    # windowSubMean = (window-np.mean(window))
    # windowStandardDeviation = np.sqrt((np.power(window-np.mean(window),2))/template_area).sum(axis=-1).sum(axis=-1)
    # templateStandardDeviation = np.sqrt((np.power(T-template_mean,2))/template_area).sum(axis=-1).sum(axis=-1)
    # ncc = ((templateSubMean*windowSubMean).sum(axis=-1).sum(axis=-1)/(windowStandardDeviation*templateStandardDeviation))

    ncc = cv2.matchTemplate(I, T, cv2.TM_CCORR_NORMED)
    #min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(ncc)

    print("Finished NCC")
    return ncc

def Cv2SSD(I,T):
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
def HarrisCorner(I, threshold:int=200, displayImage:bool=False):
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

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



#Less efficient SAD because it does not use numpy arrays
def OldSAD(I,T):
    newimage = np.copy(I)
    isColorImage = None

    imagedimensions = newimage.shape
    image_height = imagedimensions[0]
    image_width = imagedimensions[1]


    if(isinstance(I[0][0],np.ndarray) or isinstance(I[0][0],list)):
        isColorImage = True
        print("THIS IS A COLOR IMAGE")
    else:
        isColorImage = False
        print("THIS IS A GRAYSCALE IMAGE")


    centerOfKernel = -1

    templateDimensions = T.shape
    template_height = templateDimensions[0]
    template_width = templateDimensions[1]

    middleRow = math.floor(template_height/2)
    middleColumn = math.floor(template_width/2)
    kernelTotal = np.sum(a=T)

    print("SHAPE OF TEMPLATE:"+str(T.shape))

    # if(middleRow==middleColumn):
    #     centerOfKernel = middleColumn
    # else:
    #     raise Exception('The kernel is not a perfect square! ERROR')
    #     return None

    #Template Matching:
    POSITIVE_INFINTY = float('inf')
    smallestFoundSummedValues = POSITIVE_INFINTY #Grayscale image

    matchingFoundImage = 0
    num_pixels = 0
    for row_index in range(0,image_height):

        for column_index in range(0,image_width):
            # if(row_index>=280 and column_index>=380):
            #     print("ENTERED")
            #     ''

            num_pixels+=1



            #Calculates the new intensity value:
            summedKernelIntensityValues = 0 #summedValues

            summedKernelBlueValues = 0
            summedKernelGreenValues = 0
            summedKernelRedValues= 0



            rowStart = row_index
            columnStart = column_index
            rowEnd = rowStart+template_height
            columnEnd = columnStart+template_width

            currentTemplateMatrix = {
                "summedValues":0,
                "rowStart": rowStart,
                "rowEnd": rowEnd,
                "columnStart":columnStart,
                "columnEnd":columnEnd
            }


            #print(I.shape[1],I.shape[0])
            if(rowStart>=0 and rowEnd<=I.shape[0] and columnStart>=0 and columnEnd<=I.shape[1]):
                #print("ENTERED FSDFDS")
                for kernel_row_index in range(0,template_height):


                    for kernel_column_index in range(0,template_width):
                        if(isColorImage==False):
                            currentTemplateValue = T.item((kernel_row_index,kernel_column_index))
                        else:
                            currentTemplateValue = T[kernel_row_index][kernel_column_index]
                            #print(currentTemplateValue)

                        #CV2 Uses BGR array layout for colors
                        currentBluePixelValue = 0
                        currentGreenPixelValue = 0
                        currentRedPixelValue = 0


                        currentRowKernelLinedAgainstImage = row_index+kernel_row_index
                        currentColumnKernelLinedAgainstImage = column_index+kernel_column_index
                        #print("CURRENT ROW, COLUMN INDEX:"+str(row_index)+","+str(column_index))
                        #print("CURRENT KERNEL LINED AGAINST IMAGE INDEX:"+str(currentRowKernelLinedAgainstImage)+","+str(currentColumnKernelLinedAgainstImage))


                        if(currentRowKernelLinedAgainstImage>=0 and currentRowKernelLinedAgainstImage<image_height and currentColumnKernelLinedAgainstImage>=0 and currentColumnKernelLinedAgainstImage<image_width):
                            #print("LINING UP KERNEL AGAINST:"+str(currentRowKernelLinedAgainstImage)+","+str(currentColumnKernelLinedAgainstImage))
                            #currentIntensityPixelValue = I[currentRowKernelLinedAgainstImage][currentColumnKernelLinedAgainstImage]

                            if(isColorImage==False):
                                #print("ENTERED")
                                currentIntensityPixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage)
                                newIntensityValue = abs(currentTemplateValue-currentIntensityPixelValue)
                                summedKernelIntensityValues += newIntensityValue

                            elif (isColorImage):
                                currentBluePixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage,0)
                                newBluePixelValue = abs(currentTemplateValue[0]-currentBluePixelValue)
                                summedKernelIntensityValues+=newBluePixelValue

                                currentGreenPixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage,1)
                                newGreenPixelValue = abs(currentTemplateValue[1]-currentGreenPixelValue)
                                summedKernelIntensityValues += newGreenPixelValue

                                currentRedPixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage,2)
                                newRedPixelValue = abs(currentTemplateValue[2]-currentRedPixelValue)
                                summedKernelIntensityValues += newRedPixelValue



                        else:
                            ''



                            #print("COULD NOT LINE KERNEL AGAINST INVALID INDEX:"+str(currentRowKernelLinedAgainstImage)+","+str(currentColumnKernelLinedAgainstImage))


                        #print("KROW, KCOL:" + str(kernel_row_index) + "," + str(kernel_column_index) + "|CURRENT INTENSITY:" + str(currentIntensityPixelValue) + "|CURRENT KERNEL VALUE:" + str(currentKernelValue) + "|NEW INTENSITY VALUE:" + str(newIntensityValue))

                        #print("CURRENT SUMMED VALUE:"+str(summedKernelIntensityValues))

                #print(summedKernelIntensityValues)

                #Making sure the kernel total is atleast 1 (to not divide by 0 when using sobel edge kernels)
                if (kernelTotal <= 0):
                    kernelTotal = 1
                    print("ENTERED KERNEL TOTAL")


                currentTemplateMatrix['summedValues'] = summedKernelIntensityValues
                print("MATCHING FOUND IMAGE:" + str(currentTemplateMatrix))
                if(summedKernelIntensityValues<smallestFoundSummedValues):

                    smallestFoundSummedValues =summedKernelIntensityValues
                    matchingFoundImage = currentTemplateMatrix



    print(matchingFoundImage)
    return newimage

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

def LetUserChooseMatchingScore(I,T):
    useroption = ""

    selectingOption = True

    while(useroption!='exit'):
        if(selectingOption):
            print("Select which matching score technique to use:\n"
                  "1)SAD\n"
                  "2)SSD\n"
                  "3)NCC\n"
                  "(enter the digit)")
            useroption = input()

            if(useroption.isdigit()==False):
                print("Invalid option! (needs to be a digit: 1,2,3)")
            else:
                if(useroption=="1"):
                    print("You selected SAD")
                    amountofvalues = 50

                    foundSADValues = SAD(I,T)
                    smallestSADvalues = getNSmallestValues(foundSADValues, amountofvalues)
                    points = getIndexesLTEElement(foundSADValues, smallestSADvalues[amountofvalues-1])

                    displayImageGivenArray(drawRectanglesAtGivenPointsTemplate(I, points, template), windowTitle='SAD')

                elif(useroption=="2"):
                    print("You selected SSD")
                    amountofvalues = 1

                    foundSSDValues = SSD(I,T)
                    smallestSSDvalues = getNSmallestValues(foundSSDValues, amountofvalues)
                    points = getIndexesLTEElement(foundSSDValues, smallestSSDvalues[amountofvalues-1])

                    displayImageGivenArray(drawRectanglesAtGivenPointsTemplate(I, points, template), windowTitle='SSD')
                elif(useroption=="3"):
                    print("You selected NCC")
                    amountofvalues = 1

                    foundNCCValues = NCC(I,T)
                    biggestNCCvalues = getNLargestValues(foundNCCValues, amountofvalues)
                    print(biggestNCCvalues)

                    #points = getIndexesLTEElement(foundNCCValues, biggestNCCvalues[amountofvalues-1])
                    points = getIndexesForGivenElement(foundNCCValues,np.amax(foundNCCValues))

                    print("DISPLAYING")
                    displayImageGivenArray(drawRectanglesAtGivenPointsTemplate(I, points, template), windowTitle='NCC')
                break



def disparity(left,right,templateSize,window):
    image_height = left.shape[0]
    image_width = left.shape[1]
    halfTemplate = int((templateSize-1)/2)
    disparityImage = np.zeros(left.shape,dtype=np.float32)

    for row in range(halfTemplate,image_height-halfTemplate):
        smallestRow = max(row-halfTemplate,0)
        biggestRow = min(row+halfTemplate+1,image_height)

        for column in range(halfTemplate,image_width-halfTemplate):
            smallestColumn = int(max(column-halfTemplate,0))
            biggestColumn = int(min(column+halfTemplate+1,image_width))

            template = left[smallestRow:biggestRow,smallestColumn:biggestColumn].astype(np.float32)
            print(template.shape)

            disparityMin = int(max((column-window)/2,0))
            disparityMax = int(min((column+window)/2+1,image_width))

            roi = right[smallestRow:biggestRow,disparityMin:disparityMax].astype(np.float32)
            print(roi.shape)

            correspondence = Cv2SSD(roi,template)
            print("MADE")
            print(correspondence)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(correspondence)
            print(minLoc)
            disparityImage[row,column]=np.arange(correspondence.shape[1])[minLoc[0]]

    print(disparityImage)
    return disparityImage

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

def newNCC(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]



    templateSize = T.shape[1]

    halfTemplateSize = int(templateSize/2)
    #print(halfWindowSize)

    AllWindows = []

    correspondenceValues = []

    if(keepSlidingWindowWithinImage==False):
        for r in range(0,image_height,stepSize):
            for c in range(0,image_width,stepSize):

                if(r == 15 and c ==15):
                    print('f')

                blankTemp = np.zeros_like(T)

                #Ensures that the indexes are within the image
                startRow = max(0,r-halfTemplateSize)
                startColumn = max(0,c-halfTemplateSize)
                endRow = min(image_height-1,r+halfTemplateSize)
                endColumn = min(image_width-1,c+halfTemplateSize)

                # print("Start:" + str((startRow, startColumn)))
                # print("CURR:" + str((r, c))+"|"+str(I[r][c]))
                # print("End:"+str((endRow,endColumn)))

                #Using the start,end indexes, indexing the window out:
                window = I[startRow:endRow+1,startColumn:endColumn+1]
                # print("WINDOW")
                # print(window)
                AllWindows.append(window)

                cToSR = r - startRow
                cToSC = c-startColumn

                ERToc = endRow-r
                ECToc = endColumn-c

                # Ensures that the indexes are within the image
                TstartRow = max(0, halfTemplateSize-cToSR)
                TstartColumn = max(0, halfTemplateSize-cToSC)
                TendRow = min(templateSize-1, halfTemplateSize+ERToc)
                TendColumn = min(templateSize-1, halfTemplateSize+ECToc)

                # print("cToSR:" + str(cToSR))
                # print("cToSC:"+str(cToSC))
                #
                # print("ERToc:" + str(ERToc))
                # print("ECToc:" + str(ECToc))


                #ROW,COLUMN
                #STARTROW:ENDROW, STARTCOLUMN:ENDCOLUMN
                template = T[TstartRow:TendRow + 1, TstartColumn:TendColumn + 1]

                ExampleTemplate = np.array([
                    [2, 5, 5],
                    [4, 0, 7],
                    [7, 5, 9]
                ])
                ExampleImage = np.array([
                    [2, 7, 5, 8, 6],
                    [1, 7, 4, 2, 7],
                    [8, 4, 6, 8, 5]
                ])


                #print("ROW DIFF:")
                #print(endRow+1-startRow)

                # print("TEMPLATE")
                # print(template)
                # print('---')

                correspondence = np.sum(np.power(template-window,2))
                correspondenceValues.append(correspondence)

    else:
        for r in range(0, image_height, stepSize):
            for c in range(0, image_width, stepSize):
                print("CURR:" + str((r, c)))


                # Ensures that the indexes are within the image
                startRow = r
                startColumn = c
                endRow = r+templateSize
                endColumn = c+templateSize

                if(endRow<=image_height and endColumn<=image_width):

                    print("Start:" + str((startRow, startColumn)))
                    print("End:" + str((endRow, endColumn)))

                    # Using the start,end indexes, indexing the window out:
                    window = I[startRow:endRow, startColumn:endColumn]
                    print(window)
                    print('---')
                    AllWindows.append(window)

                    template  = T

                    correspondence = np.sum(np.power(template-window,2))
                    correspondenceValues.append(correspondence)
    print(correspondenceValues)
    return correspondenceValues

def newSSD(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]

    print("IMAGE SHAPE:"+str(I.shape))
    print("TEMPLATE SHAPE:"+str(T.shape))



    templateSize = T.shape[1]

    halfTemplateSize = int(templateSize/2)
    #print(halfWindowSize)

    AllWindows = []

    correspondenceValues = []

    if(keepSlidingWindowWithinImage==False):
        for r in range(0,image_height,stepSize):
            for c in range(0,image_width,stepSize):

                if(r == 15 and c ==15):
                    print('f')

                blankTemp = np.zeros_like(T)

                #Ensures that the indexes are within the image
                startRow = max(0,r-halfTemplateSize)
                startColumn = max(0,c-halfTemplateSize)
                endRow = min(image_height-1,r+halfTemplateSize)
                endColumn = min(image_width-1,c+halfTemplateSize)

                # print("Start:" + str((startRow, startColumn)))
                # print("CURR:" + str((r, c))+"|"+str(I[r][c]))
                # print("End:"+str((endRow,endColumn)))

                #Using the start,end indexes, indexing the window out:
                window = I[startRow:endRow+1,startColumn:endColumn+1]
                # print("WINDOW")
                # print(window)
                AllWindows.append(window)

                cToSR = r - startRow
                cToSC = c-startColumn

                ERToc = endRow-r
                ECToc = endColumn-c

                # Ensures that the indexes are within the image
                TstartRow = max(0, halfTemplateSize-cToSR)
                TstartColumn = max(0, halfTemplateSize-cToSC)
                TendRow = min(templateSize-1, halfTemplateSize+ERToc)
                TendColumn = min(templateSize-1, halfTemplateSize+ECToc)

                # print("cToSR:" + str(cToSR))
                # print("cToSC:"+str(cToSC))
                #
                # print("ERToc:" + str(ERToc))
                # print("ECToc:" + str(ECToc))


                #ROW,COLUMN
                #STARTROW:ENDROW, STARTCOLUMN:ENDCOLUMN
                template = T[TstartRow:TendRow + 1, TstartColumn:TendColumn + 1]

                ExampleTemplate = np.array([
                    [2, 5, 5],
                    [4, 0, 7],
                    [7, 5, 9]
                ])
                ExampleImage = np.array([
                    [2, 7, 5, 8, 6],
                    [1, 7, 4, 2, 7],
                    [8, 4, 6, 8, 5]
                ])


                #print("ROW DIFF:")
                #print(endRow+1-startRow)

                print("SSD TEMPLATE")
                print(template)
                print('---')

                print("SSD WINDOW:")
                print(window)


                correspondence = np.sum(np.power(template-window,2))
                correspondenceValues.append(correspondence)

    else:
        for r in range(0, image_height, stepSize):
            for c in range(0, image_width, stepSize):
                print("CURR:" + str((r, c)))


                # Ensures that the indexes are within the image
                startRow = r
                startColumn = c
                endRow = r+templateSize
                endColumn = c+templateSize

                if(endRow<=image_height and endColumn<=image_width):

                    print("Start:" + str((startRow, startColumn)))
                    print("End:" + str((endRow, endColumn)))

                    # Using the start,end indexes, indexing the window out:
                    window = I[startRow:endRow, startColumn:endColumn]
                    print(window)
                    print('---')
                    AllWindows.append(window)

                    template  = T

                    correspondence = np.sum(np.power(template-window,2))
                    correspondenceValues.append(correspondence)
    print(correspondenceValues)
    return correspondenceValues


def newSSD1D(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]

    print("INPUTTED IMAGE:")
    print(I)


    templateHeightSize = T.shape[0]#height
    templateWidthSize = T.shape[1] #width
    if (image_height != T.shape[0]):
        print(image_height)
        print(I)
        print(T)
        raise Exception("WINDOW AND TEMPLATE DO NOT HAVE THE SAME HEIGHT")

    #halfTemplateSize = int(templateSize/2)


    AllWindows = []

    correspondenceValues = []

    for c in range(0, image_width, stepSize):
        # print("CURR:" + str((0, c)))


        # Ensures that the indexes are within the image
        startRow = 0
        startColumn = c
        endRow = templateHeightSize
        endColumn = c+templateWidthSize

        if(endRow<=image_height and endColumn<=image_width):

            print("Start:" + str((startRow, startColumn)))
            # print("End:" + str((endRow, endColumn)))
            #
            # print("TEMPLATE:")
            # print(T)

            # Using the start,end indexes, indexing the window out:
            window = I[startRow:endRow, startColumn:endColumn]
            # print("WINDOW:")
            # print(window)
            # print('---')
            AllWindows.append(window)


            correspondence = np.sum(np.power(T-window,2))
            correspondenceValues.append(correspondence)
    #print(correspondenceValues)
    return correspondenceValues

def newSAD1D(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]

    print("INPUTTED IMAGE:")
    print(I)


    templateHeightSize = T.shape[0]#height
    templateWidthSize = T.shape[1] #width
    if (image_height != T.shape[0]):
        print(image_height)
        print(I)
        print(T)
        raise Exception("WINDOW AND TEMPLATE DO NOT HAVE THE SAME HEIGHT")

    #halfTemplateSize = int(templateSize/2)


    AllWindows = []

    correspondenceValues = []

    for c in range(0, image_width, stepSize):
        # print("CURR:" + str((0, c)))


        # Ensures that the indexes are within the image
        startRow = 0
        startColumn = c
        endRow = templateHeightSize
        endColumn = c+templateWidthSize

        if(endRow<=image_height and endColumn<=image_width):

            print("Start:" + str((startRow, startColumn)))
            # print("End:" + str((endRow, endColumn)))
            #
            # print("TEMPLATE:")
            # print(T)

            # Using the start,end indexes, indexing the window out:
            window = I[startRow:endRow, startColumn:endColumn]
            # print("WINDOW:")
            # print(window)
            # print('---')
            AllWindows.append(window)


            correspondence = np.sum(np.abs(T-window))
            correspondenceValues.append(correspondence)
    #print(correspondenceValues)
    return correspondenceValues


def newSAD(I,T,stepSize:int=1,keepSlidingWindowWithinImage:bool=False):
    image_height =I.shape[0]
    image_width =I.shape[1]



    templateSize = T.shape[1]

    halfTemplateSize = int(templateSize/2)
    #print(halfWindowSize)

    AllWindows = []

    correspondenceValues = []

    if(keepSlidingWindowWithinImage==False):
        for r in range(0,image_height,stepSize):
            for c in range(0,image_width,stepSize):

                if(r == 15 and c ==15):
                    print('f')

                blankTemp = np.zeros_like(T)

                #Ensures that the indexes are within the image
                startRow = max(0,r-halfTemplateSize)
                startColumn = max(0,c-halfTemplateSize)
                endRow = min(image_height-1,r+halfTemplateSize)
                endColumn = min(image_width-1,c+halfTemplateSize)

                print("Start:" + str((startRow, startColumn)))
                print("CURR:" + str((r, c))+"|"+str(I[r][c]))
                print("End:"+str((endRow,endColumn)))

                #Using the start,end indexes, indexing the window out:
                window = I[startRow:endRow+1,startColumn:endColumn+1]
                print("WINDOW")
                print(window)
                AllWindows.append(window)

                cToSR = r - startRow
                cToSC = c-startColumn

                ERToc = endRow-r
                ECToc = endColumn-c

                # Ensures that the indexes are within the image
                TstartRow = max(0, halfTemplateSize-cToSR)
                TstartColumn = max(0, halfTemplateSize-cToSC)
                TendRow = min(templateSize-1, halfTemplateSize+ERToc)
                TendColumn = min(templateSize-1, halfTemplateSize+ECToc)

                print("cToSR:" + str(cToSR))
                print("cToSC:"+str(cToSC))

                print("ERToc:" + str(ERToc))
                print("ECToc:" + str(ECToc))


                #ROW,COLUMN
                #STARTROW:ENDROW, STARTCOLUMN:ENDCOLUMN
                template = T[TstartRow:TendRow + 1, TstartColumn:TendColumn + 1]

                ExampleTemplate = np.array([
                    [2, 5, 5],
                    [4, 0, 7],
                    [7, 5, 9]
                ])
                ExampleImage = np.array([
                    [2, 7, 5, 8, 6],
                    [1, 7, 4, 2, 7],
                    [8, 4, 6, 8, 5]
                ])


                #print("ROW DIFF:")
                #print(endRow+1-startRow)

                print("TEMPLATE")
                print(template)
                print('---')

                correspondence = np.sum(np.abs(template-window))
                correspondenceValues.append(correspondence)

    else:
        for r in range(0, image_height, stepSize):
            for c in range(0, image_width, stepSize):
                print("CURR:" + str((r, c)))


                # Ensures that the indexes are within the image
                startRow = r
                startColumn = c
                endRow = r+templateSize
                endColumn = c+templateSize

                if(endRow<=image_height and endColumn<=image_width):

                    print("Start:" + str((startRow, startColumn)))
                    print("End:" + str((endRow, endColumn)))

                    # Using the start,end indexes, indexing the window out:
                    window = I[startRow:endRow, startColumn:endColumn]
                    print(window)
                    print('---')
                    AllWindows.append(window)

                    template  = T

                    correspondence = np.sum(np.abs(template - window))
                    correspondenceValues.append(correspondence)




    # print("All Windows:")
    # print(AllWindows)
    print(correspondenceValues)

def disparity2(left,right,templateSize:int=3,windowSize:int=5,stepSize:int=1,method='SSD'):
    image_height =left.shape[0]
    image_width =left.shape[1]

    halfTemplateSize = int(templateSize/2)
    disparityImage8 = np.zeros_like(left)
    disparityImage = np.zeros_like(left).astype(np.float32)

    for r in range(0,image_height,stepSize):
        for c in range(0,image_width,stepSize):
            print("CURR:"+str((r,c)))

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
            #template = template-np.mean(window)
            #print("SSD TEMPLATE:")
            #print(template)




            #print("SSD WINDOW")
            #print(window)

            if (template.shape[0]!=window.shape[0]):
                raise Exception("TEMPLATE HEIGHT!=WINDOW HEIGHT")

            print("COMPARING")
            #windowValues = np.array([newSAD1D(window,template,keepSlidingWindowWithinImage=True)])
            windowValues = np.array(Cv2SSD(window, template))
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(windowValues)

            print(minVal)
            print(minLoc)

            if(method.lower()=='ssd' or method.lower()=='sad'):
                correspondingXinRight = windowStart+minLoc[0]
            else:
                correspondingXinRight = windowStart+maxLoc[0]





            #print("X LEFT:"+str(c))
            #print("X RIGHT:"+str(correspondingXinRight))
            #print("MAX:")

            print("leftx:"+str(c))
            print("rightx:"+str(correspondingXinRight))

            #disparityValue = abs(correspondingXinRight-c) #Subtract right image location x by left image location x
            disparityValue = c-correspondingXinRight # Subtract right image location x by left image location x
            print("DISPARITY VALUE:" + str(disparityValue))
            disparityImage[r][c]=int(disparityValue)



    print(disparityImage)
    print(disparityImage.max())
    print(disparityImage.min())

    #disparityImage = disparityImage+abs(disparityImage.min()) #shifts disparity values so that they are all 0 minimum
    normalized = disparityImage * (1/disparityImage.max())

    np.set_printoptions(threshold=np.inf)
    print(normalized)

    return normalized


def disparityPyrmaid(left, right, levels):
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



ExampleTemplate = np.array([
                    [2, 5, 5],
                    [4, 0, 7],
                    [7, 5, 9]
                ])

ExampleTemplate2 = np.array([
                    [2, 5],
                    [4, 0]
                ])

ExampleTemplate3 = np.array([
                    [2, 5],
                    [4, 0],
                    [7, 5]
                ])


print("EXAMPLE:")

ExampleImage = np.array([
                    [2, 7, 5, 8, 6,2, 7, 2, 5, 5],
                    [1, 7, 4, 2, 7,2, 7, 4, 0, 7],
                    [8, 4, 6, 8, 5,2, 7, 7, 5, 9]
                ])

ExampleImage2 = np.array([
                    [2, 7, 5, 8, 6,2, 7, 2, 5, 5],
                    [1, 7, 4, 2, 7,2, 7, 4, 0, 7]
                ])

#print(ExampleImage[0:2,::])

#slidingWindow(ExampleImage,3,1,keepSlidingWindowWithinImage=True)

listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR, getabspaths=True)
#print(listOfImages)
stopsignimage = getImageArray(getImageFromListOfImages(listOfImages,'stopsign'),True)
streetimage = getImageArray(getImageFromListOfImages(listOfImages,'street'),True)



# print(ExampleImage[0:2,0:3])
# print("TESTING:")
# SAD(ExampleImage,ExampleTemplate)
# print(ExampleTemplate)
# print(np.std(ExampleTemplate,ddof=0))
# print("NCC")
# print(NCC(ExampleImage,ExampleTemplate))
# print()
#SSD(ExampleImage,ExampleTemplate)
#print(getIndexesForElement(testarr,np.amin(testarr)))


getSumOfAbsoluteDifferences(np.array([3,4,5]),np.array([4,5]))

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
# image = streetimage
# print("IMAGE:")
# print(image)
# imageoriginalsize = (image.shape[1],image.shape[0])
# print("Displaying your matching window image, size:"+str(imageoriginalsize))
#displayImageGivenArray(image,waitKey=0)


#Performs harris corner on the image to detect corners
#harriscornerimage = HarrisCorner(image,100,True)

#Lets the user pick which matching score they want to use:
#LetUserChooseMatchingScore(image,template)

#leftImagePath = browseImagesDialog(IMAGEDIR,'Select your left image')
#rightImagePath = browseImagesDialog(IMAGEDIR,'Select your right image')
leftImagePath = getImageFromListOfImages(listOfImages,'scene1.row3.col1')
rightImagePath = getImageFromListOfImages(listOfImages,'scene1.row3.col3')


# leftImage = getImageArray(leftImagePath,intensitiesOnly=True)
# rightImage = getImageArray(rightImagePath,intensitiesOnly=True)
leftImage = getImageArray(leftImagePath,intensitiesOnly=True).astype(np.float32)
rightImage = getImageArray(rightImagePath,intensitiesOnly=True).astype(np.float32)
leftImage = leftImage/255
rightImage = rightImage/255
print(leftImage.dtype)
print(leftImage)

#exit(0)


#displayImageGivenArray(leftImage,windowTitle='Left Image',waitKey=0)
#displayImageGivenArray(rightImage,windowTitle='Right Image',waitKey=0)

print("SAD--------")
newSSD1D(ExampleImage,ExampleTemplate,keepSlidingWindowWithinImage=True)
newSSD1D(ExampleImage2,ExampleTemplate2,keepSlidingWindowWithinImage=True)
newSSD1D(ExampleImage,ExampleTemplate3,keepSlidingWindowWithinImage=True)
#exit(0)
#newSAD(leftImage,ExampleTemplate)
#displayImageGivenArray(leftImage)
#disparityPyrmaid(leftImage,rightImage,5)
displayImageGivenArray(disparity2(leftImage,rightImage,templateSize=3,windowSize=50,stepSize=1),windowTitle='Disparity Image',waitKey=0)
#displayImageGivenArray(disparity2(leftImage,rightImage,templateSize=5,windowSize=10,stepSize=1),windowTitle='Disparity Image')