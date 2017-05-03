import numpy as np
import cv2 as cv2
from calibration import calibrate
from pipeline import abs_sobel
from pipeline import mag_sobel
from pipeline import saturation_filter
from pipeline import hue_filter
from pipeline import ang_sobel
from pipeline import perspective_transform
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

def process_image(image_to_process):
    undistort = cv2.undistort(image_to_process, mtx, dist, None, mtx)
    absx_im = abs_sobel(undistort, ksize=9, thresh=(60, 255))
    ang_im = ang_sobel(undistort, ksize=15, thresh=(0.7, 1.3))
    sat_im = saturation_filter(undistort, thresh=(155, 255))
    hue_im = hue_filter(undistort, thresh=(20, 26))

    pipeline = (absx_im | hue_im) | np.round((absx_im + ang_im + sat_im * 2 + hue_im) / 5).astype(np.uint8)
    pipeline = pipeline.astype(np.uint8)

    unwarped_pipeline, M, Minv = perspective_transform(pipeline)

    result = fit_polyline_to_lane(image_to_process, unwarped_pipeline, left_line_list, right_line_list, M, Minv)

    return result

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def fit_polyline_to_lane(orig_img, warped_pipeline, left_line_list, right_line_list, M, Minv):
    if len(left_line_list) != 0 and len(right_line_list) != 0:
        max_frames = 10
        if len(left_line_list) >= max_frames and len(right_line_list) >= max_frames:
            left_line_list.popleft()
            right_line_list.popleft()

        left_line = Line()
        right_line = Line()

        nonzero = warped_pipeline.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        minpix = 50
        midpoint = np.int(warped_pipeline.shape[1] / 2)


        left_line_list_fit =  np.mean([lline.current_fit for lline in left_line_list],axis=0)
        right_line_list_fit = np.mean([lline.current_fit for lline in right_line_list],axis=0)
        #left_line_list_rad =  np.mean([lline.radius_of_curvature for lline in left_line_list])
        #right_line_list_rad = np.mean([lline.radius_of_curvature for lline in right_line_list])

        left_lane_inds = ((nonzerox > (left_line_list_fit[0] * (nonzeroy ** 2) + left_line_list_fit[1] * nonzeroy + left_line_list_fit[2] - margin)) & \
                          (nonzerox < (left_line_list_fit[0] * (nonzeroy ** 2) + left_line_list_fit[1] * nonzeroy + left_line_list_fit[2] + margin)) & \
                          (nonzerox < midpoint))
        right_lane_inds = ((nonzerox > (right_line_list_fit[0] * (nonzeroy ** 2) + right_line_list_fit[1] * nonzeroy + right_line_list_fit[2] - margin)) & \
                           (nonzerox < (right_line_list_fit[0] * (nonzeroy ** 2) + right_line_list_fit[1] * nonzeroy + right_line_list_fit[2] + margin)) & \
                           (nonzerox > midpoint))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        ploty = np.linspace(0, warped_pipeline.shape[0] - 1, warped_pipeline.shape[0])
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit a second order polynomial to each
        # if there are new points, use those, if not, dont replace the old one
        # and just use the old fit
        if len(leftx) > minpix or len(lefty) > minpix:
            left_fit_temp = np.polyfit(lefty, leftx, 2)
            left_line.current_fit = left_fit_temp

            #temp_left_curverad = ((1 + (2 * left_fit_temp[0] * y_eval * ym_per_pix + left_fit_temp[1]) ** 2) ** 1.5) / np.absolute( 2 * left_fit_temp[0])
            #if left_line_list_rad * 0.1 < temp_left_curverad < left_line_list_rad * 10:
            left_line.current_fit = left_fit_temp
            left_line_list.append(left_line)

        if len(rightx) > minpix or len(righty) > minpix:
            right_fit_temp = np.polyfit(righty, rightx, 2)
            right_line.current_fit = right_fit_temp

            #temp_right_curverad = ((1 + (2 * right_fit_temp[0] * y_eval * ym_per_pix + right_fit_temp[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_temp[0])
            #if right_line_list_rad * 0.1 < temp_right_curverad < right_line_list_rad * 10:
            right_line.current_fit = right_fit_temp
            right_line_list.append(right_line)

        left_line_list_fit = np.mean([lline.current_fit for lline in left_line_list], axis=0)
        right_line_list_fit = np.mean([lline.current_fit for lline in right_line_list], axis=0)

        # Generate x and y values for plotting

        left_fitx = left_line_list_fit[0] * ploty ** 2 + left_line_list_fit[1] * ploty + left_line_list_fit[2]
        right_fitx = right_line_list_fit[0] * ploty ** 2 + right_line_list_fit[1] * ploty + right_line_list_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped_pipeline, warped_pipeline, warped_pipeline)) * 255
        #window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curve_rad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        left_line.radius_of_curvature = left_curve_rad
        right_line.radius_of_curvature = right_curve_rad
        # Now our radius of curvature is in meters

        y_bot = 665 * ym_per_pix
        left_front_car = left_fit_cr[0] * y_bot ** 2 + left_fit_cr[1] * y_bot + left_fit_cr[2]
        right_front_car = right_fit_cr[0] * y_bot ** 2 + right_fit_cr[1] * y_bot + right_fit_cr[2]
        mid_front_car_x = np.absolute((left_front_car + right_front_car) / 2)
        off_center = np.absolute(mid_front_car_x - np.absolute(midpoint) * xm_per_pix)
        vehicle_offset = off_center


        '''
        left_line_window = np.array([np.transpose(np.vstack([left_fitx , ploty]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_line_pts = np.hstack((left_line_window, right_line_window))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([lane_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        '''

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_pipeline).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))


        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        color_warp = cv2.addWeighted(color_warp, 1, out_img, 1, 0)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (warped_pipeline.shape[1], warped_pipeline.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(orig_img, 1, newwarp, 0.5, 0)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Radius of curvature (Left) = %.2f m' % (left_curve_rad), (10, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, 'Radius of curvature (Right) = %.2f m' % (right_curve_rad), (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, 'Vehicle position = %.2f m off center' % (vehicle_offset),(10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #plt.imshow(result)
        #plt.show()
        return result

    left_line = Line()
    right_line = Line()

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_pipeline[int(warped_pipeline.shape[0]*3/4):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_pipeline, warped_pipeline, warped_pipeline))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_pipeline.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_pipeline.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_pipeline.shape[0] - (window+1)*window_height
        win_y_high = warped_pipeline.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_line.allx = leftx
    left_line.ally = lefty
    right_line.allx = rightx
    right_line.ally = righty

    # Fit a second order polynomial to each
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    right_line.current_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped_pipeline.shape[0]-1, warped_pipeline.shape[0])
    left_fitx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
    right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

    '''
    left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_line_pts = np.hstack((left_line_window, right_line_window))
    '''


    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    left_line.radius_of_curvature = left_curve_rad
    right_line.radius_of_curvature = right_curve_rad
    # Now our radius of curvature is in meters

    y_bot = 665 * ym_per_pix
    left_front_car = left_fit_cr[0] * y_bot ** 2 + left_fit_cr[1] * y_bot + left_fit_cr[2]
    right_front_car = right_fit_cr[0] * y_bot ** 2 + right_fit_cr[1] * y_bot + right_fit_cr[2]
    mid_front_car_x = np.absolute((left_front_car + right_front_car) / 2)
    off_center = np.absolute(mid_front_car_x - np.absolute(midpoint) * xm_per_pix)
    vehicle_offset = off_center

    '''
    window_img = np.zeros_like(out_img)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([lane_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0,1280)
    plt.ylim(720,0)
    plt.show()
    '''

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_pipeline).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    color_warp = cv2.addWeighted(color_warp, 1, out_img, 1, 0)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped_pipeline.shape[1], warped_pipeline.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB), 1, newwarp, 0.5, 0)

    left_line_list.append(left_line)
    right_line_list.append(right_line)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of curvature (Left) = %.2f m' % (left_curve_rad), (10, 40), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Right) = %.2f m' % (right_curve_rad), (10, 70), font, 1, (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle position = %.2f m off center' % (vehicle_offset),(10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #plt.imshow(result)
    #plt.show()
    return result

from collections import deque

#testing with two consecutive image frames
mtx, dist = calibrate()
left_line_list = deque()
right_line_list = deque()
'''
line_im = cv2.imread('./test_images/test2.jpg')
line_im = cv2.GaussianBlur(line_im, (5,5), 0)

undistort = cv2.undistort(line_im, mtx, dist, None, mtx)
bright_const_im = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)

#line_im = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
absx_im = abs_sobel(undistort, ksize=9, thresh=(60,255))
absy_im = abs_sobel(undistort, ksize=5,orient='y', thresh=(60,255))
mag_im = mag_sobel(undistort, ksize=9, thresh=(30,255))
ang_im = ang_sobel(undistort, ksize=15, thresh=(0.7,1.3))
sat_im = saturation_filter(undistort, thresh=(155,255))
hue_im = hue_filter(undistort, thresh=(20,26))

pipeline = (absx_im | hue_im) | np.round((absx_im+ang_im + sat_im*3 + hue_im) / 6).astype(np.uint8)
pipeline = pipeline.astype(np.uint8)

warped, M, Minv = perspective_transform(undistort)
rgb_line_im = cv2.cvtColor(undistort, cv2.COLOR_BGR2RGB)
rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


warped_pipeline, M, Minv = perspective_transform(pipeline)
histogram = np.sum(warped_pipeline[warped_pipeline.shape[0]//2:,:], axis=0)

result= fit_polyline_to_lane(line_im, warped_pipeline, left_line_list,right_line_list,M, Minv)
print(len(left_line_list))
print(len(right_line_list))
result = fit_polyline_to_lane(line_im, warped_pipeline, left_line_list,right_line_list,M, Minv)
print(len(left_line_list))
print(len(right_line_list))
result = fit_polyline_to_lane(line_im, warped_pipeline, left_line_list,right_line_list,M, Minv)
print(len(left_line_list))
print(len(right_line_list))
'''

adv_output = 'adv_lane_line.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(adv_output, audio=False)

