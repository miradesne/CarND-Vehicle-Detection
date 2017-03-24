import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def loadImages(path, format = 'jpg'):
	files = glob.glob(path + '*.' + format)
	images = [mpimg.imread(file) for file in files]
	if format == 'png':
		images = np.uint8(list(map(lambda x: 255 * x, images)))
	return images

def showImgs(imgs, titles):
	count = len(imgs)
	f, axes = plt.subplots(1, count, figsize=(24, 9))
	f.tight_layout
	for i in range(len(axes)):
		img = imgs[i]
		ax = axes[i]
		if len(img.shape) == 3:
			ax.imshow(img)
		else:
			ax.imshow(img, cmap ='gray')
		ax.set_title(titles[i], fontsize=12)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()