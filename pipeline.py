import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from calibration import calibrate

def perspective_transform(img):
    src = np.float32([[256,669], [592,450], [688,450],
                      [1032,669]])

    dst = np.float32([[200, 720], [200, 0], [1000, 0],
                      [1000, 720]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    #perspective warp expects dimensions reversed
    img_size = (img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M, Minv

def abs_sobel(img, ksize=3, orient='x', thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=ksize)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    scaled = np.absolute(sobel/np.max(sobel)*255)

    zeros = np.zeros(sobel.shape, dtype=np.uint8)
    zeros[(scaled >= thresh[0])&(scaled <= thresh[1])] = 1
    return zeros

def mag_sobel(img, ksize=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    mag = np.sqrt(sobelx**2 + sobely**2)

    scaled = mag/np.max(mag)*255

    zeros = np.zeros(scaled.shape, dtype=np.uint8)
    zeros[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return zeros

def ang_sobel(img, ksize=3, thresh=(0,np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=ksize))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))

    ang = np.arctan2(sobely,sobelx)

    zeros = np.zeros(ang.shape, dtype=np.uint8)
    zeros[(ang >= thresh[0]) & (ang <= thresh[1])] = 1
    return zeros

def saturation_filter(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    zeros = np.zeros(hls.shape[0:2], dtype=np.uint8)
    zeros[(hls[:,:,2] >= thresh[0])&(hls[:,:,2] <= thresh[1])] = 1
    return zeros

def hue_filter(img, thresh=(0,255)):
    #less than 18-23 is white and yellow lane lines
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    zeros = np.zeros(hls.shape[0:2], dtype=np.uint8)
    zeros[(hls[:,:,0] >= thresh[0])&(hls[:,:,0] <= thresh[1])] = 1
    return zeros

'''
#test code and image generation code
line_im = cv2.imread('./test_images/test2.jpg')
line_im = cv2.GaussianBlur(line_im, (5,5), 0)

mtx, dist = calibrate()

undistort = cv2.undistort(line_im, mtx, dist, None, mtx)
bright_const_im = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)

f, (axarr1, axarr2) = plt.subplots(1, 2)
f.tight_layout()
axarr1.imshow(cv2.cvtColor(line_im, cv2.COLOR_BGR2RGB))
axarr1.set_title("Original Image")
axarr2.imshow(cv2.cvtColor(undistort, cv2.COLOR_BGR2RGB))
axarr2.set_title("Undistorted Image")
plt.show()

#brightness experiment
bright_const_im[:,:,1] = 128

#line_im = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
absx_im = abs_sobel(undistort, ksize=9, thresh=(60,255))
absy_im = abs_sobel(undistort, ksize=5,orient='y', thresh=(60,255))
mag_im = mag_sobel(undistort, ksize=9, thresh=(30,255))
ang_im = ang_sobel(undistort, ksize=15, thresh=(0.7,1.3))
sat_im = saturation_filter(undistort, thresh=(155,255))
hue_im = hue_filter(undistort, thresh=(20,26))

pipeline = (absx_im | hue_im) | np.round((absx_im+ang_im + sat_im*3 + hue_im) / 6).astype(np.uint8)
pipeline = pipeline.astype(np.uint8)

f, axarr = plt.subplots(2, 3)
f.tight_layout()
axarr[0,0].imshow(absx_im, cmap='gray')
axarr[0,0].set_title("Sobel X")
axarr[1,0].imshow(ang_im, cmap='gray')
axarr[1,0].set_title("Angular Sobel")
axarr[0,1].imshow(hue_im, cmap='gray')
axarr[0,1].set_title("Hue Channel")
axarr[1,1].imshow(bright_const_im)
axarr[1,1].set_title("Brightness Constant")
axarr[0,2].imshow(sat_im, cmap='gray')
axarr[0,2].set_title("Saturation Channel")
axarr[1,2].imshow(pipeline, cmap='gray')
axarr[1,2].set_title("Combined Pipeline")

plt.show()

warped, M, Minv = perspective_transform(undistort)
rgb_line_im = cv2.cvtColor(undistort, cv2.COLOR_BGR2RGB)
rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

f1, (axarr1, axarr2) = plt.subplots(1, 2)
f1.tight_layout()
axarr1.imshow(rgb_line_im)
axarr1.set_title("Original Image")
axarr2.imshow(rgb_warped)
axarr2.set_title("Perspective Transformed Image")
plt.show()

warped_pipeline, M, Minv = perspective_transform(pipeline)
histogram = np.sum(warped_pipeline[warped_pipeline.shape[0]//2:,:], axis=0)
plt.plot(histogram)
plt.show()

f1, (axarr1, axarr2) = plt.subplots(1, 2)
f1.tight_layout()
axarr1.imshow(warped_pipeline)
axarr1.set_title("Perspective Transformed Pipeline")
axarr2.imshow(rgb_warped)
axarr2.set_title("Perspective Transformed Image")
plt.show()
'''
