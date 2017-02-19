import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


# AUXILIAR FUNCTIONS FOR PROJECT 4 - ADVANCED LANE LINE DETECTION
# Calibration and distortion
def calibrate_camera(nx, ny, cal_img_files, saveimg=True, savepath='01_camera_cal'):
    #Compute camera matrix and distortion coefficients
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(cal_img_files):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file = fname.rsplit('\\',1)[-1]

        # Find the chessboard corners
        if file == 'calibration1.jpg':
            #first image 
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny-1), None)  
        else:
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points
        if ret == True:
            if file=='calibration1.jpg':
                objpoints.append(objp[:-nx,:])
            else:
                objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            if file=='calibration1.jpg':
                cv2.drawChessboardCorners(img, (nx,ny-1), corners, ret)
            else:
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            if saveimg:
                write_name = savepath + '/corners_found'+ file
                cv2.imwrite(write_name, img)
        
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist
       
# PLOT UNDISTORTED IMAGE VS ORIGINAL
def plot_chessboard(img1, img2, titles = ['','']):
    # Visualize undistorted image vs original image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(img1)
    ax1.set_title(titles[0], fontsize=12)
    ax2.imshow(img2)
    ax2.set_title(titles[1], fontsize=12);

def undistort_img_example (img, mtx, dist):
    # Example of undistorted chessboard: undistort, save to file and plot
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('01_camera_cal/test_undist.jpg',dst)
    plot_chessboard(img, dst, titles=['Original Image','Undistorted Image'])
    
def bgr_to_rgb (img):
    # Convert image from bgr to rgb color space
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
