import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt



# AUXILIAR FUNCTIONS FOR PROJECT 4 - ADVANCED LANE LINE DETECTION
# Functions used in thresholding image
def img_thresh(img, absg_thresh = (30,100), magg_thresh = (15,100), magg_kernel = 3, dirg_thresh = (0.5,1.5),
               dirg_kernel = 3, s_thresh = (100,255), plot_thresh=False):
    '''
    Function used to create a thresholded binary image
    
    Arguments:
        img: image to transform
        absg_thresh: threshold for absolute gradient
        magg_thresh: threshold for magnitude gradient
        magg_kernel: kernel to use in magnitude gradient
        dirg_thresh: threshold for direction gradient
        dirg_kernel: kernel to use in direction gradient
        s_ghresh: thershold for s channel
        plot_thresh: if True, plot image trhesholded by abs, magnitude, direction and s-channel
    return:
        (combined_binary*255, sxbinary, sxmagbinary, sxanglebinary, s_binary)
        where combined_binary*255 is thresholded final image,
        sxbinary is binary with absolute threshold
        sxmagbinary is binary with magnitude threshold
        sxanglebinary is binary with direction threshold
        s_binary is binary with s_channel threshold
    '''

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold absolute gradient
    sxbinary = abs_sobel_thresh(gray, orient='x', thresh = absg_thresh)

    # Threshold magnitude of gradient
    sxmagbinary = mag_thresh(gray, sobel_kernel=magg_kernel, mag_thresh=magg_thresh)

    # Threshold x gradient angle
    sxanglebinary = dir_threshold(gray, sobel_kernel=dirg_kernel, thresh=dirg_thresh)

    # HLS threshold
    s_binary = hls_select(img, thresh=s_thresh)
    # combined_binary[(s_binary == 1) | (gradient_binary == 1)] = 1
    
    # COMBINE THRESHOLD IMAGES
    # copy image
    combined_binary = s_binary.copy()
    
    # Combine both s_channel and absoute thresholds
    combined_binary [(sxbinary == 1)] =1
    
    #Use s_binary threshold and remove pixels that not pass direction criteria
    combined_binary[(sxanglebinary == 0)] =0
    
    if plot_thresh:
        # Plotting thresholded images
        f,((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

        ax1.set_title('sxbinary')
        ax1.imshow(sxbinary, cmap='gray')
        ax2.set_title('sxmagbinary')
        ax2.imshow(sxmagbinary, cmap='gray')
        ax3.set_title('sxanglebinary')
        ax3.imshow(sxanglebinary, cmap='gray')
        ax4.set_title('s_binary')
        ax4.imshow(s_binary, cmap='gray')
        plt.show();

    return (combined_binary*255, sxbinary, sxmagbinary, sxanglebinary, s_binary)

def abs_sobel_thresh(gray, orient='x', thresh = (0,255)):
    # Threshold absolute gradient
    thresh_min= thresh[0]
    thresh_max= thresh[1]
    #img readen using mpimg: is in RGB 
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    (x,y) =  (int (orient=='x'), int(orient=='y'))
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y)    
    abs_sobel = np.absolute(sobel) 
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Threshold magnitude of gradient
    #img readen using mpimg: is in RGB 
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2) 
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & 
                (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # HLS threshold: aply color transformation, select s channel and threshold it
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    
    #threshold s channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Threshold x gradient angle
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.abs(sobelx) 
    abs_sobely = np.abs(sobely) 
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    angle = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(sobelx)
    binary_output[(angle> thresh[0]) & 
                (angle <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def bgr_to_rgb (img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
