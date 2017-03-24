from utilities import *
from extract_features import *

import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pickle

class Classifier:
	model = LinearSVC()
	X_scaler = 0
	ystart = 400
	ystop = 656
	scale = 1.2
	spatial_size = (32, 32)
	hist_bins = 32

	def __init__(self, load_cached_features = False, visualize = True, colorspace="HLS",
				orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = "ALL"):
		self.load_cached_features = load_cached_features
		self.visualize = visualize
		self.colorspace = colorspace # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.hog_channel = hog_channel # Can be 0, 1, 2, or "ALL"

	def train(self, train_data_path):
		vehicle_path = train_data_path + "vehicles/"
		non_vehicle_path = train_data_path + "non-vehicles/"
		cars = loadImages(vehicle_path, "png")
		non_cars = loadImages(non_vehicle_path, "png")
		print("data loaded")

		car_features = []
		notcar_features = []

		# For testing: reduce the sample size
		sample_size = len(cars)

		# Parameter tweaking
		colorspace = self.colorspace 
		orient = self.orient
		pix_per_cell = self.pix_per_cell
		cell_per_block = self.cell_per_block
		hog_channel = self.hog_channel 
		spatial_size = self.spatial_size
		hist_bins = self.hist_bins

		print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')

		if self.load_cached_features == False:
			cars = cars[0:sample_size]
			non_cars = non_cars[0:sample_size]
			print("sample size: ", sample_size)

			t=time.time()

			car_features = extract_features(cars, cspace=colorspace, spatial_size=spatial_size,
									hist_bins=hist_bins, hist_range=(0, 256),
									orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
									hog_channel=hog_channel)
			notcar_features = extract_features(non_cars, cspace=colorspace, spatial_size=spatial_size,
									hist_bins=hist_bins, hist_range=(0, 256),
									orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
									hog_channel=hog_channel)

			t2 = time.time()
			print(round(t2-t, 2), 'Seconds to extract HOG features...')

			if len(car_features) == 0:
				raise Exception("there are no features!")


			car_dict = {}
			car_dict["car_features"] = car_features
			car_dict["notcar_features"] = notcar_features

			with open(train_data_path + 'features.p','wb') as f:
				pickle.dump(car_dict,f)
		else:
			car_dict = pickle.load( open( train_data_path + 'features.p', "rb" ) )
			car_features = car_dict["car_features"]
			notcar_features = car_dict["notcar_features"]

		# Create an array stack of feature vectors
		X = np.vstack((car_features, notcar_features)).astype(np.float64)                      
		# Fit a per-column scaler
		self.X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = self.X_scaler.transform(X)
		# print("shape222:", scaled_X.shape)

		if self.visualize == True:
			car_ind = np.random.randint(0, sample_size)
			# Plot an example of raw and scaled features
			fig = plt.figure(figsize=(12,4))
			plt.subplot(131)
			plt.imshow(cars[car_ind])
			plt.title('Original Image')
			plt.subplot(132)
			plt.plot(X[car_ind])
			plt.title('Raw Features')
			plt.subplot(133)
			plt.plot(scaled_X[car_ind])
			plt.title('Normalized Features')
			fig.tight_layout()
			plt.show()

		# Define the labels vector
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
		# Split up data into randomized training and test sets
		rand_state = np.random.randint(0, 100)
		x_train, x_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
		print('Feature vector length:', len(x_train[0]))

		# Use a linear SVC 
		svc = self.model
		# Check the training time for the SVC
		t=time.time()
		svc.fit(x_train, y_train)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to train SVC...')
		# Check the score of the SVC
		print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))
		# Check the prediction time for a single sample
		t=time.time()
		n_predict = 10
		print('My SVC predicts: ', svc.predict(x_test[0:n_predict]))
		print('For these',n_predict, 'labels: ', y_test[0:n_predict])
		t2 = time.time()
		print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

		self.model = svc
		return svc


