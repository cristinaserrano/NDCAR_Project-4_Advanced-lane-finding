import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from importlib import reload
from sklearn.externals import joblib

import datetime as dt

import func_calibration as custom_calib
import func_threshold as custom_th
import func_perspective as custom_perspective
import func_detect as custom_detect


# AUXILIAR FUNCTIONS FOR PROJECT 4 - ADVANCED LANE LINE DETECTION
def process_image(initial_img):
    
    #TODO load M, Minv
    mtx = joblib.load('01_camera_cal/mtx.pkl')
    dist = joblib.load('01_camera_cal/dist.pkl')
    
    #Load transformation matrix
    M = joblib.load('03_perspective_transform/M.pkl')
    Minv = joblib.load('03_perspective_transform/Minv.pkl')
    src = joblib.load('03_perspective_transform/src.pkl')
    dst = joblib.load('03_perspective_transform/dst.pkl')
    
    options = { 'absg_thresh' : (30,100),
            'magg_thresh' : (15,100),  'magg_kernel' : 3,
            'dirg_thresh' : (0.5,1.5), 'dirg_kernel' : 3,
            's_thresh' : (100,255),
            'plot_thresh': False}
    
    img = cv2.undistort(initial_img, mtx, dist, None, mtx)
    
    #obtain thresholded image (gradient & color)
    combined_binary,a,b,c,d = custom_th.img_thresh(img, **options)
    
    #perspective transform thresholded img
    warped_thresh = custom_perspective.apply_transform(combined_binary, M)
    warped_img = custom_perspective.apply_transform(combined_binary, M)

    plots=False
    img_use = 0.5
    draw_windows=False
    nwindows=15
    margin=120
    minpix=100
    img_size = (img.shape[1], img.shape [0]) 
    
    #Detect line pixels. Use histograms to detect starting position
    leftx, lefty, rightx, righty = custom_detect.detect_lane_line_pixels (warped_thresh, nwindows, margin, minpix, 
                                                               draw_windows=True, plots=plots, img_use=img_use)

    #Find points corresponding to a fitted polynomial
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fitx, right_fitx, left_curverad, right_curverad, left_curverad_cr, right_curverad_cr = custom_detect.adjust_polynomial(\
                    warped_thresh, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)
    
    # Compute car offset
    offset = custom_detect.compute_vehicle_offset(warped_thresh, leftx, lefty, rightx, righty, xm_per_pix)

    #paint lane (left line in red, right line in blue, inbetween in green)
    result = custom_detect.paint_lane(warped_thresh, leftx, lefty, rightx, righty, left_fitx, right_fitx)

    #Invert transformation of lane-painted image and merge into original image
    unwarped = cv2.warpPerspective(result, Minv, img_size, flags= cv2.INTER_LINEAR)
    final_img = cv2.addWeighted(unwarped, 1,initial_img, 1., 0)
    
    # Paint annotations in final image
    font = cv2.FONT_HERSHEY_SIMPLEX
    if (offset >=0.):
        txt = ('Vehicle is {:.2f} m right of center'.format(np.abs(offset)))
    else:
        txt = ('Vehicle is {:.2f} m left of center'.format(np.abs(offset)))

    cv2.putText(final_img, txt, (50,50),font, 1.5,(255,255,255),2,cv2.LINE_AA)

    curve = .5*left_curverad_cr + .5*right_curverad_cr
    txt = 'Radius of curvature: {:.2f} m'.format(curve)
    cv2.putText(final_img, txt, (50,100),font, 1.5,(255,255,255),2,cv2.LINE_AA)

    if False: #if true, frame images are saved
        filename='input_img/'+str(dt.datetime.now().timestamp()).replace('.','_')
        cv2.imwrite(filename+'input.jpg', pj.bgr_to_rgb(initial_img))
        cv2.imwrite(filename+'output.jpg', pj.bgr_to_rgb(final_img))
    return final_img