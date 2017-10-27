import cv2
import numpy as np
import random
import grab_cut
import imutils
# WITH OVERLAP
BACK_IMAGE_FILENAME = '/home/pratyusha/SEM_4-1/DIP/Python/img5_fore.jpg'
FORE_IMAGE_FILENAME = '/home/pratyusha/SEM_4-1/DIP/Python/img5_back.jpg'

VERBOSE = True
scalefactor = 0.5

def adjust_gamma(image, gamma=1.35):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def imresize(img,scale):
    imout = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return imout
    
# kinda corrects color
def colorcorrect(img1,img2):
    fore_image = img1
    back_image = img2
    
    # Convert to LAB colorspace
    
    fore_lab = cv2.cvtColor(fore_image, cv2.COLOR_BGR2LAB)
    back_lab = cv2.cvtColor(back_image, cv2.COLOR_BGR2LAB)
    
    # split color channels
    fore_channels = cv2.split(fore_lab)
    back_channels = cv2.split(back_lab)
    
    # generate Contrast Limited Histogram Equalization
    # equalize L channel for brightness
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    fore_histnorm = clahe.apply(fore_channels[0])
    back_histnorm = clahe.apply(back_channels[0])
    
    # put equalized channel back
    fore_channels[0] = fore_histnorm
    back_channels[0] = back_histnorm
    
    fore_LabCorr = cv2.merge(fore_channels)
    back_LabCorr = cv2.merge(back_channels)
    
    # convert back to BGR
    fore_corr = cv2.cvtColor(fore_LabCorr, cv2.COLOR_LAB2BGR)
    back_corr = cv2.cvtColor(back_LabCorr, cv2.COLOR_LAB2BGR)
    
    concat_image = np.concatenate((fore_image, fore_corr), axis=1)
    cv2.imshow('color corrected front image', imresize(concat_image,0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    concat_image = np.concatenate((back_image, back_corr), axis=1)
    cv2.imshow('color corrected back image', imresize(concat_image,0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return fore_corr, back_corr

# Draw bounding box around the person in the image
def find_bounding_box(img):
    face_cascade = cv2.CascadeClassifier(OPENCV_PATH + FACE_DETECTION_XML)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lw = list()
    rl = list()
    # detect faces
    faces = face_cascade.detectMultiScale(gray,rejectLevels=rl,levelWeights=lw, scaleFactor=1.3,minNeighbors=2)
    if len(faces) == 0:
	return None
        #x,y,w,h = rects[0]	
    if len(faces) == 1:
        x, y, w, h = faces[0]
        return (x-int(w*1),y-int(h*1), x+int(w*2),img.shape[0]), (x, y, x+w, y+h)
    if len(faces) > 1:
        # assume biggest box is the target face
        biggest_face = 0
        biggest_face_idx = 0
        for i in range(0, len(faces)):
            x, y, w, h = faces[i]
            if (w*h) > biggest_face:
                biggest_face = w*h
                biggest_face_idx = i
        x, y, w, h = faces[biggest_face_idx]
        return (x-int(w*1),y-int(h*1), x+int(w*2),img.shape[0]), (x, y, x+w, y+h)

def find_bounding_box_hog(img_hog):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img_hog, winStride=(2,2),padding=(8,8), scale=1.05, useMeanshiftGrouping=1)
    x,y,w,h = rects[-1]	
    if len(rects) == 0:
	return None
    return (x,y,w+x,h+y),(x+int(w/3),y+int(h/10),x+2*int(w/3),y+2*int(h/8))

# Return True if bounding boxes are far apart enough
# col_start, col_end are x axis values where bounding boxes do not intersect
# left_box indicates which bb is to the left of the other
def bounding_box_far(bb1, bb2, width):
    WIDTH_RATIO_THESHOLD = 0.25
    x_top_left1, y_top_left1, x_bottom_right1, y_bottom_right1 = bb1
    x_top_left2, y_top_left2, x_bottom_right2, y_bottom_right2 = bb2

    col_start = 0
    col_end = 0
    distance = 0
    left_box = 0
    if x_bottom_right1 < x_top_left2: # fore img person is left of back img
        col_start = x_bottom_right1
        col_end = x_top_left2
        distance = x_top_left2 - x_bottom_right1
        left_box = 0
    elif x_bottom_right2 < x_top_left1: # fore img person is right of back img
        col_start = x_bottom_right2
        col_end = x_top_left1
        distance = x_top_left1 - x_bottom_right2
        left_box = 1
    else: # subjects are overlapping
        return False, None, None, None

    distance_ratio = float(distance) / float(width)
    if VERBOSE: print "distance ratio between image subjects is: " + str(distance_ratio)
    if distance_ratio > WIDTH_RATIO_THESHOLD:
        return True, col_start, col_end, left_box
    else: # subjects are too close to each other
        return False, None, None, None

def alpha_blend(img_left, img_right, col_start, col_end):
    step = 1.0 / (float(col_end) - float(col_start))    
    rows, cols, channels = img_left.shape
    for r in range(0, rows):
        i = 1
        for c in range(0, cols):
            if (c < col_start):
                continue
            if (c > col_end):
                img_left[r][c] = img_right[r][c]
                continue
            img_left[r][c] = ((1-step*i)*img_left[r][c][0] + (step*i)*img_right[r][c][0], (1-step*i)*img_left[r][c][1] + (step*i)*img_right[r][c][1], (1-step*i)*img_left[r][c][2] + (step*i)*img_right[r][c][2])
            i += 1
    return img_left

def crop_image(img, H):
    rows, cols, channels = img.shape
    pts = np.float32([[0,0], [cols, 0], [cols, rows], [0, rows]]).reshape(-1,1,2)
    warp_points = cv2.perspectiveTransform(pts, H)

    top_row = int(max(max(warp_points[0][0][1], warp_points[1][0][1]), 0))
    bottom_row = int(min(min(warp_points[2][0][1], warp_points[3][0][1]), rows))

    return img[top_row:bottom_row, 0:cols]
    
# Return True if keypoint is outside of bounding box, else, return False
#   kp = (x, y), bb = (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
def keypoint_outside_bbox(kp, bb):
    x = kp.pt[0]
    y = kp.pt[1]
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = bb
    if (x>x_top_left) and (x<x_bottom_right) and (y>y_top_left) and (y<y_bottom_right):
            return False
    return True

# Find the keypoints of two images and return the keypoint matches sorted by 
# distance attribute
def find_keypoint_matches(img1, img2):
    orb = cv2.ORB()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    # this should be brute force hamming
    kp_img1, des_img1 = orb.detectAndCompute(img1,None)
    kp_img2, des_img2 = orb.detectAndCompute(img2,None)

    matches = bf.match(des_img1, des_img2)
    sorted_matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    if VERBOSE: print 'found and sorted ' + str(len(matches)) + ' matches by distance'
    return sorted_matches, kp_img1, kp_img2

# Prune matches based distance attribute
def prune_keypoint_matches_by_distance(matches):
    FACTOR = 10
    pruned_matches = []
    threshold = matches[-1].distance * FACTOR
    for m in matches:
        if m.distance < threshold:
            pruned_matches.append(m)
    if VERBOSE: print 'number of matches after distance pruning: ' + str(len(pruned_matches))
    return pruned_matches

# Prune matches that are inside bounding box. We do not want to match keypoints
# inside a bounding box because that is where the people in the image are.
def prune_keypoint_matches_by_bounding_box(matches, kp_fore, bb_fore, kp_back, bb_back):
    pruned_matches = []
    for m in matches:
        kp = kp_fore[m.queryIdx]
        if not keypoint_outside_bbox(kp, bb_fore): continue
        if not keypoint_outside_bbox(kp, bb_back): continue   # should actually be homography on bb_back
        kp = kp_back[m.trainIdx]
        if not keypoint_outside_bbox(kp, bb_back): continue
        if not keypoint_outside_bbox(kp, bb_fore): continue   # should actually be homography on bb_fore
        pruned_matches.append(m)
    if VERBOSE: 'number of matches after bounding box pruning: ' + str(len(pruned_matches))
    return pruned_matches

def calculate_homography(matches, kp_obj, kp_scene):
    obj = []
    scene = []
    for m in matches:
        obj.append(kp_obj[m.queryIdx].pt)
        scene.append(kp_scene[m.trainIdx].pt)
    H = cv2.findHomography(np.array(scene),np.array(obj), cv2.RANSAC)
    return H
    
# Make sure foreground image is the one with the bigger head
# Return True if fore img and back img need to be swapped
def swap_fore_back(bb_fore_face, bb_back_face):
    h_fore = bb_fore_face[3] - bb_fore_face[1]
    w_fore = bb_fore_face[2] - bb_fore_face[0]
    h_back = bb_back_face[3] - bb_back_face[1]
    w_back = bb_back_face[2] - bb_back_face[0]
    area_fore = w_fore*h_fore
    area_back = w_back*h_back
    
    if VERBOSE:
        print 'area of head in fore img: ' + str(area_fore)
        print 'area of head in back img: ' + str(area_back)
    
    # NOTE: back_image is actually the foreground, fore_image is background
    if area_back < area_fore:
        return True
    else:
        return False

# Blends and merges the cropped image with the background
# NOTE: background_img corresponds to the TRUE background unlike the rest of the code
def blend_cropped_image(background_img, cropped_img):
    
    height, width, channels = cropped_img.shape
    cropped_img_binary = np.zeros((height, width, 1), np.uint8)    
    
    # create binary mask
    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    cv2.threshold(cropped_img_gray, 0, 255, cv2.THRESH_BINARY, \
    cropped_img_binary)
    
    # display binary mask
    concat_image = np.concatenate((imresize(cropped_img_gray ,scalefactor), \
    imresize(cropped_img_binary,scalefactor)), axis=1)
    cv2.imshow('binarized cropped image', imresize(concat_image,0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # create a cut and paste version    
    merged_img = np.uint8((255-cropped_img_binary)/255)*background_img + \
    cropped_img
    cv2.imshow('original merged image', imresize(merged_img,0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # INSERT HERE
    # technically we should do some form of dilation or filtering
    # we can find the "black islands" in the picture and check the size of
    # these islands. If the size of the black island is small, then it means
    # that the island is actually part of the foreground (there must be black
    # in the foreground) and we should fill in the binary mask with white pxls
    
    # create structuring element
    # element_sizes is a list of tuples
    # element 1 of tuple is the erosion kernel size
    # element 2 of the tuple is the coefficient used to multiply with the
    # foreground/cropped image
    # each tuple is a consecutive iteration/erosion/blending on top of the
    # previous iteration
    element_sizes = [(5, 0)]
    outer_mask = cropped_img_binary
    for erosion_size, fore_coeff in element_sizes:
        element_size = (erosion_size, erosion_size)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, element_size)
        
        # erode foreground/cropped part
        inner_mask = np.zeros((height, width, 1), np.uint8)
        cv2.erode(outer_mask, element, dst=inner_mask)
        
        # display image and eroded image
        concat_image = np.concatenate((imresize(outer_mask,scalefactor), \
        imresize(inner_mask,scalefactor)), axis=1)
        cv2.imshow('outer mask vs. inner mask', concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # create mask
        mask = outer_mask - inner_mask
        
        # display mask
        cv2.imshow('mask', imresize(mask,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('mask', imresize(mask/255*cropped_img,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        blended_section = np.uint8(fore_coeff*mask/255*cropped_img + \
        (1-fore_coeff)*mask/255*background_img)
        
        cv2.imshow('blended mask', imresize(blended_section,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # create inverse mask
        inverse_mask = 255 - mask
        inverse_mask = inverse_mask/255 # convert to 1s and 0s
        merged_img = inverse_mask*merged_img + blended_section
        
        cv2.imshow('merged image', imresize(merged_img,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # update outer mask for next iteration
        outer_mask = inner_mask
        
    rows, cols, channels = inner_mask.shape
    for i in range(0,rows):
        for j in range(0,cols):
            if inner_mask[i,j] == np.uint8(255):
                bottom_row = i
                break

    print 'bottom row: ' + str(bottom_row)
    cv2.imshow('final mask', imresize(inner_mask,scalefactor))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    merged_img = merged_img[0:bottom_row, 0:cols]
    
    cv2.imshow('final result', imresize(merged_img,scalefactor))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return merged_img

####### Display Image Functions #########

# Print lines that connect matching keypoints across two images
def display_keypoint_matches(img1, img2, kp1, kp2, matches):
    concat_image = np.concatenate((img1, img2), axis=1)
    h, w = img1.shape[:2]
    for m in matches:
        color = (0, 255, 0)
        cv2.line(concat_image, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), (int(kp2[m.trainIdx].pt[0] + w), int(kp2[m.trainIdx].pt[1])), color)
    cv2.imshow('keypoint matches', imresize(concat_image,scalefactor))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Print circles over keypoints on two images
def display_keypoints(img1, img2, kp1, kp2):
    for kp in kp1:
        cv2.circle(img1, (int(kp.pt[0]), int(kp.pt[1])), 2, (255,0,0))
    for kp in kp2:
        cv2.circle(img2, (int(kp.pt[0]), int(kp.pt[1])), 2, (255,0,0))

    concat_image = np.concatenate((img1, img2), axis=1)
    cv2.imshow('keypoints', imresize(concat_image,scalefactor))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# foreground image will be the query image/object
# background image will be the train image/scene
if __name__ == '__main__':

    import local_config as config

    # set parameters
    OPENCV_PATH = config.opencv['OPENCV_PATH']
    FACE_DETECTION_XML = config.opencv['FACE_DETECTION_XML']

    # read images
    fore_image = cv2.imread(FORE_IMAGE_FILENAME)
    back_image = cv2.imread(BACK_IMAGE_FILENAME)
    fore_image_hog = imutils.resize(fore_image, width=min(800, fore_image.shape[1]))
    back_image_hog = imutils.resize(back_image, width=min(800, back_image.shape[1]))
     
    # color correction
    fore_image, back_image = colorcorrect(fore_image,back_image)
    algo_flag = ''
    # find body bounding box
    bb_fore,bb_fore_face = find_bounding_box(fore_image)
    bb_back,bb_back_face = find_bounding_box(back_image)
    if bb_fore_face == None or bb_back_face == None:
    	bb_fore,bb_fore_face = find_bounding_box_hog(fore_image_hog)
    	bb_back,bb_back_face = find_bounding_box_hog(back_image_hog)
    	fore_image = adjust_gamma(fore_image_hog)
    	back_image = adjust_gamma(back_image_hog)
	algo_flag = "hog"
    # swap fore and back images if necessary (fore has bigger body)    
    if swap_fore_back(bb_fore_face, bb_back_face):
        tmp = fore_image # due to shallow copy
        fore_image = back_image
        back_image = tmp
       
        tmp = bb_fore_face
        bb_fore_face = bb_back_face
        bb_back_face = tmp
       
        tmp = bb_fore # due to shallow copy
        bb_fore = bb_back
        bb_back = tmp
    
    # show bounding boxes of bodies
    import copy # can move this if it bothers you...just wanted to keep it together
    fore_image_temp = copy.deepcopy(fore_image)
    cv2.rectangle(fore_image_temp, (bb_fore[0], bb_fore[1]), (bb_fore[2], bb_fore[3]), (0, 255, 0), 3)
    cv2.rectangle(fore_image_temp, (bb_fore_face[0], bb_fore_face[1]), (bb_fore_face[2], bb_fore_face[3]), (0, 255, 0), 3)
    back_image_temp = copy.deepcopy(back_image)
    cv2.rectangle(back_image_temp, (bb_back[0], bb_back[1]), (bb_back[2], bb_back[3]), (0, 255, 0), 3)
    cv2.rectangle(back_image_temp, (bb_back_face[0], bb_back_face[1]), (bb_back_face[2], bb_back_face[3]), (0, 255, 0), 3)
    cv2.imshow('color corrected fore image with bounding box', fore_image_temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('color corrected back image with bounding box', back_image_temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    concat_image = np.concatenate((fore_image_temp, back_image_temp), axis=1)
    cv2.imshow('color corrected images with bounding boxes', concat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rows, cols, channels = fore_image.shape
    far, col_start, col_end, left_box = bounding_box_far(bb_fore, bb_back, cols)
    if far:
        if VERBOSE: print 'Image subjects are far apart, can use panorama flow'
        # treat as a 2 picture panoramic
        matches, kp_fore, kp_back = find_keypoint_matches(fore_image, back_image)
        matches = prune_keypoint_matches_by_distance(matches)
        matches = prune_keypoint_matches_by_bounding_box(matches, kp_fore, bb_fore, kp_back, bb_back)
        display_keypoint_matches(fore_image, back_image, kp_fore, kp_back, matches)
        H, mask = calculate_homography(matches, kp_fore, kp_back)
        warped_back = cv2.warpPerspective(back_image, H, (fore_image.shape[:2][1], fore_image.shape[:2][0]))

        concat_image = np.concatenate((fore_image, warped_back), axis=1)
        cv2.imshow('warped back image', imresize(concat_image,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # do alpha blending
        if left_box:    # figure out which is to the left of the other
            warped_back, fore_image = fore_image, warped_back
            # bb_fore, bb_back = bb_back, bb_fore
        blended_image = alpha_blend(fore_image, warped_back, col_start, col_end)
        cv2.imshow('blended image', imresize(blended_image,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cropped_image = crop_image(blended_image, H)
        cv2.imshow('cropped image', imresize(cropped_image,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # need to do segmentation
        if VERBOSE: print 'Image subjects too close together, need to do segmentation.'
        matches, kp_fore, kp_back = find_keypoint_matches(fore_image, back_image)
        # display_keypoints(fore_image, back_image, kp_fore, kp_back)

        matches = prune_keypoint_matches_by_distance(matches)
        matches = prune_keypoint_matches_by_bounding_box(matches, kp_fore, bb_fore, kp_back, bb_back)
        display_keypoint_matches(fore_image, back_image, kp_fore, kp_back, matches)

        H, mask = calculate_homography(matches, kp_fore, kp_back)
        warped_back = cv2.warpPerspective(back_image, H, (fore_image.shape[:2][1], fore_image.shape[:2][0]))

        concat_image = np.concatenate((fore_image, warped_back), axis=1)
        cv2.imshow('warped back image', imresize(concat_image,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
	if algo_flag == "hog":
        	warped_back_cut = grab_cut_hog.grab_cut(warped_back)
	else:
        	warped_back_cut = grab_cut.grab_cut(warped_back)
        blended_image = blend_cropped_image(fore_image, warped_back_cut)

        cv2.imshow('blended image', imresize(blended_image,scalefactor))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# if they are close together, then do grab cut
    # blending is harder in this case -> think about it more
    # maybe do some sort of averaging for pixels that overlap in image
