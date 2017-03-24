**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/training_imgs.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/output1.png
[image5]: ./output_images/output2.png
[image6]: ./output_images/output3.png
[image7]: ./output_images/frame1.png
[image8]: ./output_images/frame5.png
[image9]: ./output_images/frame6.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 38 through 96 of the file called `extract_features.py`, lines 32 through 71 in `train_model.py`, and lines 6 through 11 in `utilities.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  I defined a function called `loadImages()` in `utitlities.py` that takes in a folder path and image type. If the image type is "png", I scale the image from 0-1 to 0-255 so they match with the jpg images. 

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I started from orientation 6 and settled down on 9 because it has all common directions. I started from RGB, but switched to HLS because the channels have more distinct features while RGB has redundant features between the 3 channels. I also tried other various combinations of parameters and different color spaces to check which one could have the best accuracy. I tested them fast on a smaller sample size (500) of training data by splitting the train/test data to 0.8/0.2. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I created a linear SVM using sklearn.svm library. Then I trained the classifier by using HOG features on all channels, spatial features, and color histogram features. The code that I trained the model is defined in a function called `train()` in the class `Classifier` in `train_model.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search code is defined as a function called `find_cars()` in `sliding_window.py`. The function takes in the image, y start position, stop position, scale, the classifier, scaler and all the parameters I used to extract the HOG and other features during training. 

I started from using a 1.5 scaled of 64 by 64 windows. I also chose to only search within a range of vertical coordinates (starting from 400 and stoping at 656), because above that area it's sky and trees where the cars would not show up.

I changed the scale to 1.3 because it's big enough to have features in a window, but not too big that there would be multiple cars in one window. We want multiple windows on a car so we can apply the heat map threshold to get rid of some false positive windows.

I chose an overlap of 0.5 so each window is overlapped by another one. 

Here's an example of the windows.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on scale 1.3 using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also use heat map to remove some false positive points.

Here are some example output images with heat maps:

![alt text][image4]
![alt text][image5]
![alt text][image6]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I store the heatmap into an array of heat maps for the most 5 recent images. Then for each frame I take the average of the heatmaps, and then further threshold the average to produce the output by using `scipy.ndimage.measurements.label()`.

### Here are three frames as the results of `scipy.ndimage.measurements.label()` and their corresponding heatmaps:

![alt text][image7]

![alt text][image8]

![alt text][image9]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think one issue is sliding windows. Since the sliding windows capture only the partial of the images, it can have a lot of noise and sometimes we cannot find the perfect scales of the windows. Multiple scales can be used, but small scales would result in a lot of noises, while big scales would miss some of the detections. The time to extract HOG features is extremely long, and sliding windows can potentially be very consuming too if the scale is small and the number of windows is huge. 

What I can see from my training is that the classifier can still have a bunch of false-positive on the trees/shades even though the test accuracy is 99%. I'm not sure how to prevent the classifier from overfitting other than adding more training data. It seems that using HOG is very time consuming to make the model more general.

I think deep learning can probably make the model more robust. I'm tuning a lot of parameters and it feels that I'm hard coding what features I want the model to learn. Perhaps it's better to use deep learning and let the model find out features by itself. Maybe some noise reduction before training the classifier will help with the performance too.

