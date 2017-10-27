import numpy as np
import cv2

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
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img, winStride=(2, 2),
                                            padding=(8, 8), scale=1.05,
                                            useMeanshiftGrouping=1)
    x, y, w, h = rects[-1]
    rect = (x, y, min(x + w, img.shape[0]), min(y + h, img.shape[1]))
    mask = np.zeros(img.shape[:2], np.uint8)
    for c in xrange(x - 2 * w, x + 2 * w):
        for r in xrange(y + int(h), img.shape[0]):
            mask[r][c] = cv2.GC_PR_FGD

    for c in xrange(x, x + w):
        for r in xrange(int(y - 0.5 * h), y + h):
            mask[r][c] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img
