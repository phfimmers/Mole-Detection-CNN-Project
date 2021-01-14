'''This file contains the preprocessing function applied
to all images train/test/uploads'''

import cv2
from math import ceil
import matplotlib.pyplot as plt

def selma_secret_sauce(image, advanced = True, bigger_border = False, replicate_border = False, for_training = True, model_size = (256, 256)):
    '''Makes a square image out of the input image (COLOR_BGR).
    Options:
    for_training: if True, do not resize the image to model_size, because it
                will be done after data augmentation by ImageDataGenerator.
    bigger_border: can be used for training give more space for rotation in augmentation
    replicate_border: if true, edge pixels are copied in the border in stead of
                        using GREY
    '''

    def crop(image):
        '''this function cuts of the small black borders found on training images,
        so that expanding to squares with pixel replication works'''
        cut_left = 2
        cut_right = 2
        cut_top = 2
        cut_bottom = 2
        top_left = (cut_left, cut_top)
        bottom_right = (image.shape[1]-cut_right, image.shape[0]-cut_bottom)
        image = image[top_left[1]:(bottom_right[1] + 1), top_left[0]:(bottom_right[0] + 1)]
        return image
        
    
    def make_square(image, skip_cropping = False):
        '''make square with blurred border replication'''
        if skip_cropping == False:
            image = crop(image)
        if bigger_border == True:
            # enlarge border size to allow more room for augmentation rotation/shifting
            square_dimension = ceil((image.shape[1]**2 + image.shape[0]**2)**0.5*1.1)
        else:
            square_dimension = max(image.shape[1],image.shape[0])
        top_border = (square_dimension-image.shape[0])//2
        bottom_border = square_dimension-image.shape[0]-top_border
        left_border = (square_dimension-image.shape[1])//2
        right_border = square_dimension-image.shape[1]-left_border
        if replicate_border == True:
            square = cv2.copyMakeBorder(image,
                                        top_border,
                                        bottom_border,
                                        left_border,
                                        right_border,
                                        cv2.BORDER_REPLICATE)
            # blur the border (and the whole image)
            square = cv2.GaussianBlur(square,(19,19),cv2.BORDER_DEFAULT)
            # impose unblurred image on blurred square
            yoff = top_border
            xoff = left_border
            square[yoff:yoff+image.shape[0], xoff:xoff+image.shape[1]] = image
        else:
            # fill the border with grey instead
            square = cv2.copyMakeBorder(image,
                                        top_border,
                                        bottom_border,
                                        left_border,
                                        right_border,
                                        cv2.BORDER_CONSTANT,
                                        value = (125, 125, 125))
        
        return square
    
    # convert BMP to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # make square
    square = make_square(image)
    
    # don't continue processing when advanced is False, and resize to model dimension
    # when predicting
    if advanced == False:
        if for_training == False:
            square = cv2.resize(square, model_size, interpolation = cv2.INTER_AREA)
        return square
    
    # convert image to grayScale
    grayScale = cv2.cvtColor(crop(image), cv2.COLOR_RGB2GRAY)

    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    hair_removal_image = cv2.inpaint(crop(image),threshold,1,cv2.INPAINT_TELEA)
    hair_removal_image = cv2.medianBlur(hair_removal_image,5)
    
    #-----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(hair_removal_image, cv2.COLOR_RGB2LAB)
    
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # make square and resize in case of prediction
    final = make_square(final, skip_cropping = True)
    if for_training == False:
        final = cv2.resize(final, model_size, interpolation = cv2.INTER_AREA)
    return final

    #_____END_____#

