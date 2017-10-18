"""
Processing utilities
"""
import cv2


def imresize(im, scale):
    """
    Resize the given image to given scale
    :param im: image to be resized
    :param scale: output scale
    :return: scaled image
    """
    imout = cv2.resize(im, (int(scale * im.shape[1]), int(scale * im.shape[0])))
    return imout


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
