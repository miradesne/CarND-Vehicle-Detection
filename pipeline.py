import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from train_model import *
from sliding_window import *
from utilities import *
from heat_map import *
import pickle

preload = True

if preload == False: 
    # if we don't load a trained model:
    classifier = Classifier(load_cached_features = True, colorspace="HLS", visualize=False)
    # extract features from the training data and train the svm classifier
    svc = classifier.train("training_data/")
    with open('classifier.p', 'wb') as f:
        # save the trained model for future usage.
        pickle.dump(classifier, f)
else:
    classifier = pickle.load( open( 'classifier.p', "rb" ) )


class Car(object):
    # a class to save the recent heat maps
    def __init__(self):
        self.n = 5
        self.recent_heat_maps = np.array([])
        self.average_heat = np.array([])
        self.detected = False

    def add_heat(self, heat_map):
        if self.recent_heat_maps.shape[0] == 0:
            self.recent_heat_maps = np.array([heat_map])
        else:
            self.recent_heat_maps = np.append(self.recent_heat_maps, [heat_map], axis=0)
            # print(self.recent_heat_maps.shape)
            self.average_heat = np.average(self.recent_heat_maps, axis=0)

        if self.recent_heat_maps.shape[0] > self.n:
            self.recent_heat_maps = np.delete(self.recent_heat_maps, 0, 0)

        

def pipeline(img, classifier, car):
    classifier.scale = 1.3
    # use sliding window to find the bounding boxes.
    output, box_list = find_cars(img, classifier.ystart, classifier.ystop, classifier.scale, classifier.model, classifier.X_scaler,
                classifier.orient, classifier.pix_per_cell, classifier.cell_per_block, 
                classifier.spatial_size, classifier.hist_bins)

    # extract heat map and apply threshold
    output, heat = get_heat_threshed(img, box_list, threshold = 1, vis=True)
    # add the heat map to the array
    car.add_heat(heat)

    # print(car.recent_heat_maps.shape[0])
    if car.recent_heat_maps.shape[0] >= car.n:
        # apply another threshold to the moving average
        smoothed_heat = apply_threshold(car.average_heat, 2)
        heatmap = np.clip(smoothed_heat, 0, 255)
        # create bounding boxes of the heat map
        labels = label(heatmap)
        output = draw_labeled_bboxes(np.copy(img), labels)

    return output


car = Car()
def process_video_img(img):
    return pipeline(img, classifier, car)

# test_img_path = "test_images/"
# imgs = loadImages(test_img_path, "jpg")
# outputs = [pipeline(img, classifier, car) for img in imgs]

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

white_output = 'test_output.mp4'
clip1 = VideoFileClip('test_video.mp4')
white_clip = clip1.fl_image(process_video_img) 
white_clip.write_videofile(white_output, audio=False)

print("done.")