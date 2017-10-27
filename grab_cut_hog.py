import numpy as np
import cv2
import copy

# get images
# get keypoints for images
# match keypoints in area where people are not standing
# calculate homography
# warp perspective
# face detect/people detect to fit bounding box on second image
# blend two images together -> if pixel values close enough, use first image
# ??
# profit
#OPENCV_PATH = '../../opencv/' 
#FACE_DETECTION_XML = 'data/haarcascades/haarcascade_frontalface_default.xml'
#IMAGE_FILENAME = 'img/old.jpg'
#filename = 'img1.jpg'

def grab_cut(img):

    import local_config as config

    OPENCV_PATH = config.opencv['OPENCV_PATH']
    FACE_DETECTION_XML = config.opencv['FACE_DETECTION_XML']    
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img, winStride=(2,2),padding=(8,8), scale=1.05, useMeanshiftGrouping=1)
    x,y,w,h = rects[-1]
#    rect = (x,y,w+x,h+y),(x+int(w/3.5),y+int(h/20),x+3*int(w/4.5),y+2*int(h/7.5))
#    face_cascade = cv2.CascadeClassifier(OPENCV_PATH + FACE_DETECTION_XML)
#    #img = cv2.imread(filename)
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
#    if len(faces) == 0: 
#        print 'no faces detected in grab_cut.py'
#        return img
#
#    x, y, w, h =  faces[0]
#    rect = (max(0,x-2*w), 0, min(x+3*w, img.shape[1]), img.shape[0])
    mask = np.zeros(img.shape[:2],np.uint8)
    
    img_temp = copy.deepcopy(img)
    cv2.rectangle(img_temp, (x, y), (x+w, y+h), (0, 255, 0), 3) #for body
    cv2.rectangle(img_temp, (x+int(w/3.5),y+int(h/20)),(x+3*int(w/4.5),y+2*int(h/7.5)), (0, 255, 0), 3)
    cv2.imshow('bb for grabcut', img_temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rect = (x,y,min(x+w,img.shape[0]),min(y+h,img.shape[1]))
    # cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD for background, forground and probable
    for c in xrange(x,min(x+w,img.shape[1])):
        for r in  xrange(y,min(y+h,img.shape[0])):
            mask[r][c] = cv2.GC_PR_FGD
            
    for c in xrange(x+int(w/3.5), x+3*int(w/4.5)):
        for r in xrange(y+int(h/20), y+2*int(h/7.5)):
            mask[r][c] = cv2.GC_FGD

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    # cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img
#grab_cut(filename)
