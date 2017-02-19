import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


# AUXILIAR FUNCTIONS FOR PROJECT 4 - ADVANCED LANE LINE DETECTION
# Functions for perspective transform
def apply_transform(img, M):
    #Apply transform
    img_size = (img.shape[1], img.shape [0])
    return cv2.warpPerspective(img, M, img_size, flags= cv2.INTER_LINEAR)

def perspective_transform(points, img, combined_binary):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    x = [p[0] for p in [p1,p2,p3,p4,p1]]
    y = [p[1] for p in [p1,p2,p3,p4,p1]]

    print ('Plot points to use for perspective transformation')
    # Plotting images
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.plot(x,y,'r-',linewidth = 2.)
    plt.show();

    src = np.float32([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]])
    #dst = np.float32([[p1[0], p1[1]], [p1[0], 0], [p4[0], 0], [p4[0], p1[1]]])
    dst = np.float32([[p1[0], 720], [p1[0], 0], [p4[0], 0], [p4[0], 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return M, Minv, src, dst

def plot_transformed_img(warped_rgb, warped_thresh, dst):
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    x = [x[0] for x in dst]
    y = [x[1] for x in dst]

    ax1.imshow(warped_rgb)
    ax1.plot(x,y,'r-',linewidth = 2.)

    ax2.imshow(warped_thresh, cmap='gray')
    ax2.plot(x,y,'r-',linewidth = 2.)
    plt.show();

        
def bgr_to_rgb (img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)