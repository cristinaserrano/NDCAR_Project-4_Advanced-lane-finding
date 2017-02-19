import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt



# AUXILIAR FUNCTIONS FOR PROJECT 4 - ADVANCED LANE LINE DETECTION

    
# LANE DETECTION
def _extract_x_y(lane_inds, img):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
    return x, y    

def _locate_one_window(img, out_img, window, margin, window_height, x_current,nonzerox, nonzeroy, draw_windows=True):
    #locate pixels in one window
    # Identify window boundaries in x and y (and right and left)
    win_y_low = img.shape[0] - (window+1)*window_height
    win_y_high = img.shape[0] - window*window_height
    win_x_low = x_current - margin
    win_x_high = x_current + margin
    # Draw the windows on the visualization image
    if draw_windows:
        cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
    # Append these indices to the lists
    return good_inds      

def _locate_all_windows(img, out_img, nwindows, margin, minpix,  x_current, nonzerox, nonzeroy, draw_windows=True):
    #iterate through all windows and apply _locate_one_window function
    #lane indices
    lane_inds = []
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    
    # Step through the windows one by one
    for window in range(nwindows):
        good_inds = _locate_one_window(img, out_img, window, margin, window_height, x_current, nonzerox, nonzeroy, draw_windows)
        
        #save lane indices
        lane_inds.append(good_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    return lane_inds#, out_img    

def locate_with_histogram(img, plots=True, img_use=0.5):
    #use only img_use% of lower image for histagram
    img_to_hist = img[int(img.shape[0]*(1-img_use)):,:]
    midx = np.int(img.shape[1]/2)

    histogram = np.sum(img_to_hist, axis=0)
    if plots:
        fig = plt.figure(figsize=(12,2))
        plt.plot(histogram)
        plt.title('Distribution of white pixels in lower {:.0f}% of image'.format(img_use*100))
        plt.ylabel('number of white pixels')
        plt.xlabel('image "x" coordinate')
        plt.xlim(0, 1280)        
        plt.show();
        
    leftx_base = np.argmax(histogram[:midx])
    rightx_base = np.argmax(histogram[midx:]) + midx
    if plots:
        print ('Left x starting position: {}, right x starting position: {}'.format(leftx_base, rightx_base))
    return leftx_base, rightx_base

def detect_lane_line_pixels (img, nwindows, margin, minpix, draw_windows=True, plots=True, img_use=0.4):
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #starting pixels position (at base of image)
    leftx_base, rightx_base = locate_with_histogram(img, plots, img_use=img_use)

    ### LEFT LANE
    left_lane_inds = _locate_all_windows(img, out_img, nwindows, margin, minpix, leftx_base, nonzerox, nonzeroy, draw_windows)
    # Extract left line pixel positions
    leftx, lefty =  _extract_x_y(left_lane_inds, img)

    ### RIGHT LANE
    right_lane_inds = _locate_all_windows(img, out_img, nwindows, margin, minpix, rightx_base, nonzerox, nonzeroy, draw_windows)
    # Extract left line pixel positions
    rightx, righty =  _extract_x_y(right_lane_inds, img)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if plots:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        plt.imshow(out_img)
        # plt.plot(leftx, lefty,'g.')
        plot_title = 'red/blue: points detected'
        if draw_windows:
            plot_title = plot_title + ', green: windows'
        plt.title(plot_title)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show();

    return leftx, lefty, rightx, righty

def adjust_polynomial(img, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix):
    #Fit 2nd order polynomial. Compute offset and curvature.
    img_shapex = img.shape[1]
    img_shapey = img.shape[0]

    # Obtain coefficients for a  second order polynomial to each lane line
    # in pixels
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # in meters
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shapey-1, img_shapey )    
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    #radius in pixels
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # radius in meters
    left_curverad_cr = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_cr = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    #obtain x for each y
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, left_curverad, right_curverad, left_curverad_cr, right_curverad_cr
    
def plot_detected_pixels_and_line(img, leftx, lefty, rightx, righty, left_fitx, right_fitx):
    #colour left and right line pixels
    out_img = np.dstack((img, img, img))*255
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

    #show image. Plot fitted polynomials in yellow
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.title('fitted 2nd order polylnomials in yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0);
    
def compute_vehicle_offset(img, leftx, lefty, rightx, righty, xm_per_pix):
    right_lane_base = (rightx[np.argmax(righty)])
    left_lane_base = (leftx[np.argmax(lefty)])

    vehicle_center = img.shape[1]/2
    lane_center = (right_lane_base + left_lane_base)/2
    return (vehicle_center - lane_center)*xm_per_pix

def paint_lane(img, leftx, lefty, rightx, righty, left_fitx, right_fitx):
    out_img = np.zeros_like(np.dstack((img, img, img))*255)
    #out_img[lefty, leftx] = [255, 0, 0]
    #out_img[righty, rightx] = [0, 0, 255]
    window_img = np.zeros_like(out_img)
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )      

    left_lane_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_lane_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_pts = np.hstack((left_lane_window, right_lane_window))
    
    #create green polynomial and merge into image (transparency of 0.3)
    cv2.fillPoly(window_img, np.int_([lane_pts]), (0,255, 0))
    result_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result_img