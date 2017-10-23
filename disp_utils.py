"""
Image Display Functions
"""
import cv2
import numpy as np
import copy
from main import *
from process_utils import imresize


def show(name, img):
    """
    shows the given image
    :param name: name on the windows
    :param img: image to be shown
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_bounding_boxes(fore_image, back_image, bb_fore, bb_fore_face, bb_back,
                        bb_back_face):
    """
    show bounding boxes of bodies
    :param fore_image:  foreground image
    :param back_image: background image
    :param bb_fore: bounding box for foreground
    :param bb_fore_face: face bounding box for foreground
    :param bb_back: bounding box for background
    :param bb_back_face: face bounding box for background
    """
    fore_image_temp = copy.deepcopy(fore_image)
    cv2.rectangle(fore_image_temp, (bb_fore[0], bb_fore[1]),
                  (bb_fore[2], bb_fore[3]), (0, 255, 0), 3)
    cv2.rectangle(fore_image_temp, (bb_fore_face[0], bb_fore_face[1]),
                  (bb_fore_face[2], bb_fore_face[3]), (0, 255, 0), 3)
    back_image_temp = copy.deepcopy(back_image)
    cv2.rectangle(back_image_temp, (bb_back[0], bb_back[1]),
                  (bb_back[2], bb_back[3]), (0, 255, 0), 3)
    cv2.rectangle(back_image_temp, (bb_back_face[0], bb_back_face[1]),
                  (bb_back_face[2], bb_back_face[3]), (0, 255, 0), 3)
    show('color corrected fore image with bounding box', fore_image_temp)
    show('color corrected back image with bounding box', back_image_temp)
    concat_image = np.concatenate((fore_image_temp, back_image_temp), axis=1)
    show('color corrected images with bounding boxes', concat_image)


def display_keypoint_matches(img1, img2, kp1, kp2, matches):
    """
    Print lines that connect matching keypoints across two images
    :param img1: input image1
    :param img2: input image2
    :param kp1: keypoints in image1
    :param kp2: keypoints in image2
    :param matches: matches between two keypoints
    """
    concat_image = np.concatenate((img1, img2), axis=1)
    h, w = img1.shape[:2]
    for m in matches:
        color = (0, 255, 0)
        cv2.line(concat_image,
                 (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
                 (int(kp2[m.trainIdx].pt[0] + w), int(kp2[m.trainIdx].pt[1])),
                 color)
        show('keypoint matches', imresize(concat_image, scale_factor))


def display_keypoints(img1, img2, kp1, kp2):
    """
    show keypoints in both the images
    :param img1: input image1
    :param img2: input image2
    :param kp1: keypoints in image1
    :param kp2: keypoints in image2
    """
    for kp in kp1:
        cv2.circle(img1, (int(kp.pt[0]), int(kp.pt[1])), 2, (255, 0, 0))
    for kp in kp2:
        cv2.circle(img2, (int(kp.pt[0]), int(kp.pt[1])), 2, (255, 0, 0))

    concat_image = np.concatenate((img1, img2), axis=1)
    show('keypoints', imresize(concat_image, scale_factor))
