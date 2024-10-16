import numpy as np
from scipy.ndimage import gaussian_filter
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt #to display images
import matplotlib
matplotlib.rcParams['figure.figsize'] = [5, 5]

def attack(img, sigma, th1, th2, n_layer):

    redBlur = selective_blur(img[:,:,0], sigma, th1, th2)
    greenBlur = selective_blur(img[:,:,1], sigma, th1, th2)
    blueBlur = selective_blur(img[:,:,2], sigma, th1, th2)

    redBlur = layers_blur(redBlur, sigma, n_layer)
    greenBlur = layers_blur(greenBlur, sigma, n_layer)
    blueBlur = layers_blur(blueBlur, sigma, n_layer)

    #Merge the three components
    result = cv2.merge([redBlur,greenBlur,blueBlur])

    #Blur last three layer
    result = result & 254

    return result

def selective_blur(img, sigma, th1, th2):

    # Blur the image
    blurred = gaussian_filter(img, sigma)

    #Canny detection for detecting the edges
    mask = canny_edge_detection(img, th1, th2)

    # Combination of the blur with the mask given by the canny detection
    return np.where(mask, blurred, img)

def canny_edge_detection(img, th1, th2):
    d=3 # gaussian blur

    matplotlib.rcParams['figure.figsize'] = [5, 5]

    edgeresult=img.copy()
    edgeresult = cv2.GaussianBlur(edgeresult, (2*d+1, 2*d+1), -1)[d:-d,d:-d]

    #gray = cv2.cvtColor(edgeresult, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(img, th1, th2)

def layers_blur(img, sigma, n_layer):

    blurred = gaussian_filter(img, sigma)

    tot = 0
    for i in range(n_layer):
        tot += 2**i

    b2 = blurred & tot

    t2 = 255 - tot
    b3 = b2 | t2

    img = img & b3

    return img