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

def getIndexesForElement(array,element):
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
    print(I)
    print(topLeftPoint)
    print(dimension)

    I = cv2.rectangle(I,topLeftPoint,dimension,(255,0,0),2)
    return I

def drawRectangleOnImageGivenTemplate(I,topLeftPoint,template):
    I=drawRectangleOnImage(I,topLeftPoint,(topLeftPoint[0]+template.shape[1],topLeftPoint[1]+template.shape[0]))
    return I

def SSD(I,T):
    template_height = T.shape[1]
    template_width = T.shape[0]

    view =np.lib.stride_tricks.as_strided(I,shape=(I.shape[0]-template_height+1,I.shape[1]-template_width+1,template_height,template_width),strides=I.strides*2)
    ssd = ((view-T)*(view-T)).sum(axis=-1).sum(axis=-1)
    print(ssd)
    return ssd

def SAD2(I,T):
    template_height = T.shape[1]
    template_width = T.shape[0]

    view =np.lib.stride_tricks.as_strided(I,shape=(I.shape[0]-template_height+1,I.shape[1]-template_width+1,template_height,template_width),strides=I.strides*2)
    ssd = abs(view-T).sum(axis=-1).sum(axis=-1)
    print(ssd)
    return ssd


#Less efficient SAD
def SAD(I,T):
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


ExampleTemplate = np.array([
    [2, 5, 5],
    [ 4 ,0, 7 ],
    [7, 5, 9]
])

ExampleImage = np.array([
    [2 ,7, 5, 8, 6],
    [1, 7, 4, 2, 7],
    [8, 4, 6, 8, 5]
])

print(ExampleImage[0:2,0:3])
print(SAD2(ExampleImage,ExampleTemplate))
print(SSD(ExampleImage,ExampleTemplate))
#print(getIndexesForElement(testarr,np.amin(testarr)))


getSumOfAbsoluteDifferences(np.array([3,4,5]),np.array([4,5]))

#Sets up tkinter root windows and closed it (only need it for user browsing)
root = tkinter.Tk()
root.withdraw()

print("Browse and pick a template image")
templatepath = browseImagesDialog(MessageForUser='Select your Template (Foreground)')
template = getImageArray(templatepath,True)
templateOriginalSize = (template.shape[1],template.shape[0])
print("Displaying your template image, size:"+str(templateOriginalSize))

displayImageGivenArray(template, windowTitle='Template:')


#Method1: Sliding Template T(x,y) across an Image I(x,y)
#We then want to get 'matches' for the template on the image itself
print("Enter a desired size for your template image:")
templateSize = getUserInputDimension()
template = ScaleByGivenDimensions(template,templateSize)
#displayImageGivenArray(template,windowTitle='Template(resized):')

print("\n\n")

print("Pick your matching window image:")
imagepath = browseImagesDialog(MessageForUser='Select your Template (Foreground)')
image = getImageArray(imagepath,True)
imageoriginalsize = (image.shape[1],image.shape[0])
print("Displaying your matching window image, size:"+str(imageoriginalsize))
#displayImageGivenArray(image)


foundSADValues = SSD(image,template)
minvalue = np.amin(foundSADValues)
print("MIN")
points = getIndexesForElement(foundSADValues,minvalue)
print(points[0])

displayImageGivenArray(drawRectangleOnImageGivenTemplate(image,points[0],template))
