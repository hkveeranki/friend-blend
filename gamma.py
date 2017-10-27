
# import the necessary packages
import numpy as np
import cv2
 
def imresize(img,scale):
    imout = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return imout

# kinda corrects color
def colorcorrect(img1):
    fore_image = img1
    
    # Convert to LAB colorspace
    
    fore_lab = cv2.cvtColor(fore_image, cv2.COLOR_BGR2LAB)
    
    # split color channels
    fore_channels = cv2.split(fore_lab)
    
    # generate Contrast Limited Histogram Equalization
    # equalize L channel for brightness
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    fore_histnorm = clahe.apply(fore_channels[0])
    
    # put equalized channel back
    fore_channels[0] = fore_histnorm
    
    fore_LabCorr = cv2.merge(fore_channels)
    
    # convert back to BGR
    fore_corr = cv2.cvtColor(fore_LabCorr, cv2.COLOR_LAB2BGR)
    
    #concat_image = np.concatenate((fore_image, fore_corr), axis=1)
    #cv2.imshow('color corrected front image', imresize(concat_image,0.5))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return fore_corr

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
# construct the argument parse and parse the arguments
# load the original image
img = 'img3.jpg'
original = cv2.imread(img)
adjusted2 = colorcorrect(original)
# loop over various values of gamma
for gamma in np.arange(0.75, 1.5, 0.10):
	# ignore when gamma is 1 (there will be no change to the image)
	if gamma == 1:
		continue
 
	# apply gamma correction and show the images
	gamma = gamma if gamma > 0 else 0.1
	adjusted = adjust_gamma(original, gamma=gamma)
	adjusted3 = colorcorrect(adjusted)
	adjusted4 = adjust_gamma(adjusted2,gamma=gamma)
	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Images", np.hstack([imresize(original,0.5), imresize(adjusted,0.5),imresize(adjusted2,0.5),imresize(adjusted4,0.5)]))
	cv2.waitKey(0)

