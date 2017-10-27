"""
Processing utilities
"""
import cv2
import numpy as np

def imresize(im, scale):
    """
    Resize the given image to given scale
    :param im: image to be resized
    :param scale: output scale
    :return: scaled image
    """
    imout = cv2.resize(im, (int(scale * im.shape[1]), int(scale * im.shape[0])))
    return imout


def adjust_gamma(image, gamma=1.35):
    """
    Apply gamma correction
    :param image: input image
    :param gamma: gamma value
    :return: corrected image
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# do a possible color correction
def colorcorrect(img1, img2):
    """
    Color correct the given images and return the corrected images
    :param img1: input image1
    :param img2: input image2
    :return: color corrected images
    """
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
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

    # concat_image = np.concatenate((fore_image, fore_corr), axis=1)
    # show('color corrected front image', concat_image)
    # concat_image = np.concatenate((back_image, back_corr), axis=1)
    # show('color corrected back image', concat_image)

    return fore_corr, back_corr


def swap_fore_back(bb_fore_face, bb_back_face):
    """
    Make sure foreground image is the one with the bigger head
    :param bb_fore_face: bounding box of foreground face
    :param bb_back_face: bounding box of background face
    :return: True if fore img and back img need to be swapped False otherwise 
    """
    h_fore = bb_fore_face[3] - bb_fore_face[1]
    w_fore = bb_fore_face[2] - bb_fore_face[0]
    h_back = bb_back_face[3] - bb_back_face[1]
    w_back = bb_back_face[2] - bb_back_face[0]
    area_fore = w_fore * h_fore
    area_back = w_back * h_back

    # NOTE: back_image is actually the foreground, fore_image is background
    if area_back < area_fore:
        return True
    else:
        return False
