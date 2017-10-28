import numpy as np
import cv2
import sys
import copy


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def maskk(imgg,x,y,w,h):
    img2=imgg
    for r in xrange(0,img.shape[0]):
        for c in xrange(0,img.shape[1]):
            if c>x and c<x+w and r>y and r<y+h:
                img2[r][c][:]=imgg[r][c][:]
            else:
                img2[r][c][:]=0
    return img2

def largestt(faces):
    if len(faces) == 0:
        return 0,-1,-1,-1,-1,-1
    if len(faces) == 1:
        x, y, w, h = faces[0]
    if len(faces) > 1:
        largest_face = 0
        largest_face_idx = 0
        for i in range(0, len(faces)):
            x, y, w, h = faces[i]
            if (w * h) < largest_face:
                largest_face = w * h
                largest_face_idx = i
        x, y, w, h = faces[largest_face_idx]
    return 1,w*h,x,y,w,h

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        
def draw_detections1(img, x,y,w,h):
    # the HOG detector returns slightly larger rectangles than the real objects.
    # so we slightly shrink the rectangles to get a nicer output.
    pad_w, pad_h = int(0.15*w), int(0.05*h)
    cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), 1)


def find_bounding_box_new_v(found_filtered,max_index, mxx,myy,mww,mhh,img):
    rx, ry,rw, rh = found_filtered[max_index]
    #rect = (max(0, x - 2 * w), 0, min(x + 3 * w, img.shape[1]), img.shape[0])
    rect = (rx, ry,rw, rh)
    mask = np.zeros(img.shape[:2], np.uint8)
    # cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD for background, forground and probable
    for c in xrange(rx, rx+rw):
        for r in xrange(ry,ry+rh):
            mask[r][c] = cv2.GC_PR_FGD

    for c in xrange(mxx, mxx + mww):
        for r in xrange(int(myy - 0.5 * mhh), myy + mhh):
            mask[r][c] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    mask2 = np.zeros(img.shape[:2], np.uint8)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    box = mask2.nonzero()
    a = min(box[1])
    b = max(box[1])
    c = max(box[0])
    d = min(box[0])
    img = img * mask2[:, :, np.newaxis]
    #cv2.imshow('img',img)
    #cv2.waitKey()
    return (a,c,b,d), (mxx, myy, mxx + mww, myy + mhh)

def better_bounding_box(img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    
    #draw_detections(img, found)
    #draw_detections(img, found_filtered, 3)
    #print('%d (%d) found' % (len(found_filtered), len(found)))
    #cv2.imshow('img', img)
    #ch = cv2.waitKey()
    
    #make mask from the boxes
    #then detect faces
    #choose the box with largest face
    #return box and face dimentions
    index = 0
    max_area = -1
    
    for x,y,w,h in found_filtered:
        # apply mask
        tmp = copy.deepcopy(img)
        img1 = maskk(tmp,x,y,w,h)
        face_cascade = cv2.CascadeClassifier("/home/vinamra/Downloads/opencv-master/" + "data/haarcascades/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 2)
        print len(faces)
        #now we have list of face dimentions for a single body in the image
        #choose the largest one and update max
        fla,area,xx,yy,ww,hh = largestt(faces)
        if area > max_area:
            max_area = area
            max_index = index
            mxx = xx
            myy = yy
            mww = ww
            mhh = hh
            flag= fla
        index += 1
    #now target box is found_filtered[max_index] and sure features rect is m__
    #now give this to grabcut. Remove haar from grabcut code. make sure flag is 1
    
    if flag == 1:
        imgg=copy.deepcopy(img)
        bounding_box= find_bounding_box_new_v(found_filtered,max_index, mxx,myy,mww,mhh,imgg)
        return bounding_box