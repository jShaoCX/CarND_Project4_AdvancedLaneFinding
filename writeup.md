**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./ReportFiles/Calibration_Images.png "Calibration Images"
[image2]: ./ReportFiles/Undistort_Test_Image.png "Undistort Image"
[image3]: ./ReportFiles/Pipeline_Series_Image.png "Pipeline Images"
[image4]: ./ReportFiles/Color_Image_Perspective.png "Perspective Change"
[image5]: ./ReportFiles/Histogram.png "Histogram"
[image6]: ./ReportFiles/First_Image_Polyline_Find.png "First Polyline Find"
[image7]: ./ReportFiles/Second_Image_Polyline_Find.png "Second Polyline Find"
[image8]: ./ReportFiles/Video_Image.png "Video Images"
[image9]: ./ReportFiles/First_Image_Polyline_Blend.png "Perspective Restore"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README:

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./calibration.ipynb" lines 7 to 32.

The calibration images given have 9 columns and 6 columns of corners. The object points are the chessboard coordinates in the world coordinate system. The image is a flat chessboard picture on a wall so I am assuming the z axis for all points are 0 and that the x,y coordinates of the corners are just an evenly spaced grid. The mgrid function in numpy takes care of this. The image points are found by the findChessboardCorners function in opencv. This finds the actual intersection coordinates x,y of the two white squards and two black squares. If 9 columns and 6 rows of corners are found, then the image points are appended to a list. The set of object points will also be added at that time. Multiple calibration images must be used in order come up with sensible distortion coefficients and camera matrix. The resulting coeffients and matrix from the calibrateCamera function in opencv can then be used to undistort other images via the undistort function. This corrects and distortion artifacts that the camera has (ex: if the camera has a fish eye effect, the undistort will flatten the image). Below are some examples of undistorting calibration images:

![alt text][image1]

### Pipeline (single images)

The pipeline related functions are located in "./pipeline.ipynb" lines 7 to 73.

#### 1. Provide an example of a distortion-corrected image.

The distortion correction is more subtle with actual test images of scenery. The difference is not as immediately obvious as with the calibration images but an example is shown below:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combindation of sobel filters and color space channel thresholding to come up with an optimal pipeline for the test images provided. I started with a sobel filter in the x direction and a saturation threshold but I found that this was too simplistic. The sobel filter in the x direction did not capture much of the lane line at all for the bridge images. The saturation thresholding performed very well on the left lane line but the right lane line was usually very sparse. The pipeline I ultimately arrived at was the following, located on line 104:
```python
    absx_im = abs_sobel(undistort, ksize=9, thresh=(60, 255))
    ang_im = ang_sobel(undistort, ksize=15, thresh=(0.7, 1.3))
    sat_im = saturation_filter(undistort, thresh=(155, 255))
    hue_im = hue_filter(undistort, thresh=(20, 26))

    pipeline = (absx_im | hue_im) | np.round((absx_im + ang_im + sat_im * 2 + hue_im) / 5).astype(np.uint8)
    pipeline = pipeline.astype(np.uint8)
```
It seems a little too complicated and probably could be simplified but my reasoning was the following. The sobel x performed very well on asphault and laneline and captures the outline of the lane line, hue thresholding was very specifically tuned to the lane colors so the center of each lane line can be captured very clearly. Neither would produce too much noise outside of those features so when combined, the mask would have a thick lane line (center and outline highlighted). The averaging of masks was my attempt to make a filter with features that I could control the weights of. The angular sobel filter is extremely noisy but captures the lane lines heavily along with some desirable artifacts between lane line breaks. The saturation thresholding does not capture much outside of the lane lines but is very sparse when capturing the right lane lines so I wanted to weigh the clean image a bit heavier but still grab some of the noise around the lanes. The combination was just to see if I could capture more of a cloud of points around each lane line instead of just a scattering of a few points. The following are examples of my pipeline and some of its steps:

![alt text][image3]

The brightess image was my experiment to hold brightness constant and analyze the saturation and hue channels. The actual use of the pipeline is in "./project.ipynb" lines 12 to 26.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform code is in the "./pipeline.ipynb" lines 7 to 24. The function takes in an image and had hardcoded values for the source and destination perspectives. I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([[256,669], [592,450], [688,450],
                      [1032,669]])

    dst = np.float32([[200, 720], [200, 0], [1000, 0],
                      [1000, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 256,669       | 200, 720      |
| 592,450       | 200, 0        |
| 688,450		| 1000, 0       |
| 1032,669      | 1000, 720     |

The following image pair shows the perspective transform of one of the test images:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I made use of the methods described in the lesson as seen in "./project.ipynb" lines 202 to 252. For the first frame, using the mask from the pipeline described in the previous section, I took a histogram of the bottom 3/4 of the mask. The y axis is the number of points in the bin of each x axis point across the image:

![alt text][image5]

This gave me a location of where to start looking for points. The data across the rest of the histogram (outside the windows around the histogram peaks) are just noise that need to be filtered out for a sensible line fit. Therefore, the image was divided into several rows and a window of 200 pixels was chosen for each lane line of each row. The windows are the green rectangles in the example shown below:

![alt text][image6]

After these windows are established, only points within these windows will be added to a list for a numpy polyfit. The points that were included are colored red for the left lane and blue for the right lane. There is a small amount of noise in this image at the bottom left corner. This noise is colored in white and that is an indication that it is not included in the calculation of the line. A second order polynomial was used and is shown as the yellow line.

The proceeding images used the previous polyline to establish a window of points of interest. There is no need to provide another histogram for the window search because the window is just 100 pixels out from each line from the previous image as seen in "./project.ipynb" lines 52 to 86. This method is displayed in the image below:

 ![alt text][image7]

 In order to make the fits smoother across frames, I used a deque and stored up to 10 frames of information in the Line class given by the lesson. The method just uses an average of the past 10 frames' lines and uses that as the pixel of interest window. After polylines are fit to the current image, in "./project.ipynb" lines 115 to 121, it is added to the deque and the actual fit is an average of the current line and the previous lines. This way, no one bad frame fit can create an image that looks that bad, it will be averaged out and only contribute a small amount of its poor fit.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in "./project.ipynb" lines 130 to 141. I used the methods described in the lesson. This involved getting the polynomial coefficients, y value furthest from the vehicle and the radius of curvature equation. The values seemed to be sensible. They hovered around the 500 m range during the curves in the highway, but during straight aways, it expanded out to 1000 - 4000 m. These values are within the ranges given by the lesson. The centering of the vehicle calculation is located in "./project.ipynb" lines 144 to 149. I simply took the bottom y value of the image (closest to the car) and calculated the midpoint between the two polylines of left and right lane. Then I just took the difference between the midpoint calculated and the midpoint of the actual image. The values seemed to be fairly reasonable as well (within the 1.5 to 0.1 range).

 ![alt text][image8]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To warp the prespetive back to the road in the camera image, I just used the inverted matrix of the original warp in "./project.ipynb" lines 170 to 185. Here is an example of my result on a test image. This is the first time the algorithm is run so the green rectangles used in the sliding window search are visible and are transformed along with the rest of the image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=l63cdaAtps8)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are two major components of the project that required parameter adjustment and additional code. The first is of course the pipeline creation. The individual filters needed to be tested and their thresholds found. However, these, at the moment are static thresholds. The sobel y and sobel magnitude filters did not perform particularly well for this project. The sobel x did not provide any additional utility outside of the first lane line finding project. It would still fail when driving across the concrete colored bridge. Changing the colorspace and filtering by saturation was useful but considering that some of the right lane line was a broken line and included some tiny reflectors, the saturation around the smaller white pieces was low. The left line was solid so there was less areas where the lane would be lower in yellow saturation. I included a hue filter as well to just pick up white and yellow, a very small window, but this could prove very fragile because if the color were to change slightly, no lane lines would be detected.

The second major component is smoothing out the lines across multiple frames and disregarding drastic changes in polyline fits due to sudden artifacts in the road. This was addressed by holding onto the past 10 or so frames and using the previous polyline location to start the search for a lane line and averaging the current fit with those of the past 10 or so frames. This would ensure that any sudden drastic changes to the fit would be averaged out by the previous good fits. This is definitely not perfect because if the video recievces an entire 3-4 seconds of bad fits, all of the good fit frames would have left the deque and only bad frames would be left. To remedy this I tried an approach to gate the entry to the deque with a check on the current calculated curvature of the line fit. If the line fit was completely off, then I was planning to use just the previous fit. However, while running tests on this strategy, I noticed that I had to expand the window to an entire magnitude of the best fit curvature. I did not get to the bottom of why this issue came up but will attempt to address it when I have more time. My theory was that if I just disregard an entire bad section (say the concrete bridge), then the fit would just recover when good (asphault section) frames come back up again. This approach has its faults as well because if the bridge includes a slight turn, the lane line finder would simply ignore the turn and crash the vehicle.

Ultimately, it comes down to a fairly robust pipeline. As with some strategies I tried in the first project, there must be a way to detect what colors or values are actually present in the road and continuously filter those values out. Also, if there is a way to adjust the pipeline's thresholds dynamically, then it may be possible to account for different types of roads. It may be possible to look ahead to the horizon of the road, further away, use that section to adjust the filter thresholds before the car gets close enough to truly need that information for steering guidance.
