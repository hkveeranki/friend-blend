import numpy as np
import cv2
import local_config as config

def grab_cut(img):
    """
    - get keypoints for images
    - match keypoints in area where people are not standing
    - calculate homography
    - warp perspective
    - face detect/people detect to fit bounding box on second image
    - blend two images together -> if pixel values close enough, use first image
    :param img: input image 
    :return: processed image
    """
    OPENCV_PATH = config.opencv['OPENCV_PATH']
    FACE_DETECTION_XML = config.opencv['FACE_DETECTION_XML']
    face_cascade = cv2.CascadeClassifier(OPENCV_PATH + FACE_DETECTION_XML)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    if len(faces) == 0: 
        print 'no faces detected in grab_cut.py'
        return img
    x, y, w, h =  faces[0]
    rect = (max(0,x-2*w), 0, min(x+3*w, img.shape[1]), img.shape[0])
    mask = np.zeros(img.shape[:2],np.uint8)
    for c in xrange(x-2*w, x+2*w):
        for r in xrange(y+int(h), img.shape[0]):
            mask[r][c] = cv2.GC_PR_FGD
            
    for c in xrange(x, x+w):
        for r in xrange(int(y-0.5*h), y+h):
            mask[r][c] = cv2.GC_FGD

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img