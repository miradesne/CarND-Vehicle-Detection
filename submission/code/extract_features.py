import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import glob

def convert_color(img, color="HSV"):
	if color == "RGB":
		return np.copy(img)
	colors = {"YCrCb": cv2.COLOR_RGB2YCrCb,
				"LUV": cv2.COLOR_RGB2LUV,
				"HLS": cv2.COLOR_RGB2HLS,
				"HSV": cv2.COLOR_RGB2HSV,
				"YUV": cv2.COLOR_RGB2YUV} 
	if color not in colors:
		raise Exception('invalid colorspace!!')
	return cv2.cvtColor(img, colors[color])

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
						vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
								  visualise=vis, feature_vector=feature_vec)
		plt.imshow(hog_image)
		plt.show()
		return features
	# Otherwise call with one output
	else:      
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
					   visualise=vis, feature_vector=feature_vec)
		return features

# Define a function to extract both color and HOG features
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
						hist_bins=32, hist_range=(0, 256),
						orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
						visualize=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
		# Read in each one by one
		# apply color conversion if other than 'RGB'
		# Apply bin_spatial() to get spatial color features
		# Apply color_hist() to get color histogram features
		# Append the new feature vector to the features list
	# Return list of feature vectors
	for image in imgs:
		# apply color conversion if other than 'RGB'
		feature_image = convert_color(image, cspace)     
		# Apply bin_spatial() to get spatial color features
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		# Apply color_hist() also with a color space option now
		hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
		# Append the new feature vector to the features list
		# print(spatial_features.dtype, hist_features.dtype)

		# Call get_hog_features() and append the features
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:,:,channel], 
									orient, pix_per_cell, cell_per_block, 
									vis=visualize, feature_vec=True))
			hog_features = np.ravel(hog_features)        
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
						pix_per_cell, cell_per_block, vis=visualize, feature_vec=True)

		features.append(np.concatenate((spatial_features, hist_features, hog_features)))

		if visualize == True:
			visualize = False
	# Return list of feature vectors
	return features

