import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from os import listdir

def calibrate():
    nx = 9
    ny = 6
    dir = listdir('./camera_cal')

    calib_ims = []
    im_pts = []
    obj_pts = []

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    im = cv2.imread('./camera_cal/calibration1.jpg')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    for name in dir:
        im = cv2.imread('./camera_cal/'+name)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret:
            im_pts.append(corners)
            obj_pts.append(objp)
            calib_ims.append(im)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, im_pts, gray.shape[::-1], None,None)
    return mtx, dist

'''
#create images for report
dir = listdir('./camera_cal')
nx = 9
ny = 6
calib_ims = []
for name in dir:
    im = cv2.imread('./camera_cal/' + name)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        calib_ims.append(im)

mtx, dist = calibrate()

undistort_ims = []
for distorted in calib_ims:
    undistorted = cv2.undistort(distorted, mtx, dist, None, mtx)
    undistort_ims.append(undistorted)

f, axarr = plt.subplots(2,2)
f.tight_layout()
axarr[0, 0].imshow(calib_ims[10])
axarr[0, 0].set_title("Calibration2 Original")
axarr[0, 1].imshow(undistort_ims[10])
axarr[0, 1].set_title("Calibration2 Undistorted")
axarr[1, 0].imshow(calib_ims[12])
axarr[1, 0].set_title("Calibration3 Original")
axarr[1, 1].imshow(undistort_ims[12])
axarr[1, 1].set_title("Calibration3 Undistorted")
plt.show()
#end tests
'''
