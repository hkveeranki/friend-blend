import cv2
import numpy as np
import grab_cut
import local_config as config

from process_utils import colorcorrect, imresize

VERBOSE = True
scale_factor = 1

OPENCV_PATH = config.opencv['OPENCV_PATH']
FACE_DETECTION_XML = config.opencv['FACE_DETECTION_XML']


def find_bounding_box(img):
    """
    Find the bounding box in the given image
    :param img: input image
    :return: details of the bounding box
    """
    face_cascade = cv2.CascadeClassifier(OPENCV_PATH + FACE_DETECTION_XML)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # lw = list()
    # rl = list()

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 2)

    if len(faces) == 0:
        return None
    if len(faces) == 1:
        x, y, w, h = faces[0]
        return (x - int(w * 1), y - int(h * 1), x + int(w * 2), img.shape[0]), (
            x, y, x + w, y + h)
    if len(faces) > 1:
        # assume biggest box is the target face
        biggest_face = 0
        biggest_face_idx = 0
        for i in range(0, len(faces)):
            x, y, w, h = faces[i]
            if (w * h) > biggest_face:
                biggest_face = w * h
                biggest_face_idx = i
        x, y, w, h = faces[biggest_face_idx]
        return (x - int(w * 1), y - int(h * 1), x + int(w * 2), img.shape[0]), (
            x, y, x + w, y + h)


# Return True if bounding boxes are far apart enough
# col_start, col_end are x axis values where bounding boxes do not intersect
# left_box indicates which bb is to the left of the other
def bounding_box_far(bb1, bb2, width):
    width_ratio_threshold = 0.2
    width_ratio_min = 0.05
    x_top_left1, y_top_left1, x_bottom_right1, y_bottom_right1 = bb1
    x_top_left2, y_top_left2, x_bottom_right2, y_bottom_right2 = bb2

    col_start = 0
    col_end = 0
    distance = 0
    left_box = 0
    if x_bottom_right1 < x_top_left2:  # fore img person is left of back img
        col_start = x_bottom_right1
        col_end = x_top_left2
        distance = x_top_left2 - x_bottom_right1
        left_box = 0
    elif x_bottom_right2 < x_top_left1:  # fore img person is right of back img
        col_start = x_bottom_right2
        col_end = x_top_left1
        distance = x_top_left1 - x_bottom_right2
        left_box = 1
    else:  # subjects are overlapping
        return 0, None, None, None

    distance_ratio = float(distance) / float(width)
    if VERBOSE:
        print "distance ratio between image subjects is: " + str(distance_ratio)
    if distance_ratio > width_ratio_threshold:
        return 1, col_start, col_end, left_box
    elif distance_ratio > width_ratio_min:
        return 0, None, None, None
    else:  # subjects are too close to each other
        return -1, None, None, None


def alpha_blend(img_left, img_right, col_start, col_end):
    step = 1.0 / (float(col_end) - float(col_start))
    rows, cols, channels = img_left.shape
    for r in range(0, rows):
        i = 1
        for c in range(0, cols):
            if c < col_start:
                continue
            if c > col_end:
                img_left[r][c] = img_right[r][c]
                continue
            img_left[r][c] = (
                (1 - step * i) * img_left[r][c][0] + (step * i) *
                img_right[r][c][
                    0],
                (1 - step * i) * img_left[r][c][1] + (step * i) *
                img_right[r][c][
                    1],
                (1 - step * i) * img_left[r][c][2] + (step * i) *
                img_right[r][c][
                    2])
            i += 1
    return img_left


def crop_image(img, H):
    """
    crop the image as per the homography projection
    :param img: image to be cropped
    :param H: Homography transform details
    :return: cropped image
    """
    rows, cols, channels = img.shape
    pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]).reshape(-1,
                                                                           1, 2)
    warp_points = cv2.perspectiveTransform(pts, H)
    top_row = int(max(max(warp_points[0][0][1], warp_points[1][0][1]), 0))
    bottom_row = int(min(min(warp_points[2][0][1], warp_points[3][0][1]), rows))

    return img[top_row:bottom_row, 0:cols]


def keypoint_outside_bbox(kp, bb):
    """
    Check if keypoint is outside box
    :param kp: co ordinates of keypoint(x, y) 
    :param bb: bounding box details (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
    :return: True if keypoint is outside of bounding box, else False
    """
    x = kp.pt[0]
    y = kp.pt[1]
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = bb
    return not ((x > x_top_left) and (x < x_bottom_right)
                and (y > y_top_left) and (y < y_bottom_right))


def find_keypoint_matches(img1, img2):
    """
    Find the keypoints of two images and return the keypoint matches
    :param img1: input image1 
    :param img2: input image2
    :return: keypoint matches sorted by distance attribute
    """
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,
                       crossCheck=True)  # this should be brute force hamming
    kp_img1, des_img1 = orb.detectAndCompute(img1, None)
    kp_img2, des_img2 = orb.detectAndCompute(img2, None)

    matches = bf.match(des_img1, des_img2)
    sorted_matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    if VERBOSE:
        print 'found and sorted ' + str(
            len(matches)) + ' matches by distance'
    return sorted_matches, kp_img1, kp_img2


def prune_keypoint_matches_by_distance(matches):
    """
    Prune matches based distance attribute
    :param matches: input matches
    :return: pruned matches
    """
    FACTOR = 10
    pruned_matches = []
    threshold = matches[-1].distance * FACTOR
    for m in matches:
        if m.distance < threshold:
            pruned_matches.append(m)
    if VERBOSE:
        print 'number of matches after distance pruning: ' + str(
            len(pruned_matches))
    return pruned_matches


def prune_keypoint_matches_by_bounding_box(matches, kp_fore, bb_fore, kp_back,
                                           bb_back):
    """
    Prune matches that are inside bounding box. We do not want to match keypoints
    inside a bounding box because that is where the people in the image are. 
    :param matches: keypoint smatches
    :param kp_fore: key points in foreground
    :param bb_fore: bounding box of foreground
    :param kp_back: key points the background
    :param bb_back: bounding box of background
    :return: pruned key point matches
    """
    pruned_matches = []
    for m in matches:
        kp = kp_fore[m.queryIdx]
        if not keypoint_outside_bbox(kp, bb_fore): continue
        if not keypoint_outside_bbox(kp,
                                     bb_back): continue  # should actually be homography on bb_back
        kp = kp_back[m.trainIdx]
        if not keypoint_outside_bbox(kp, bb_back): continue
        if not keypoint_outside_bbox(kp,
                                     bb_fore): continue  # should actually be homography on bb_fore
        pruned_matches.append(m)
    if VERBOSE:
        print 'number of matches after bounding box pruning: ' + str(
            len(pruned_matches))
    return pruned_matches


def calculate_homography(matches, kp_obj, kp_scene):
    """
    Caluclate the homography matchings
    """
    obj = []
    scene = []
    for m in matches:
        obj.append(kp_obj[m.queryIdx].pt)
        scene.append(kp_scene[m.trainIdx].pt)
    H = cv2.findHomography(np.array(scene), np.array(obj), cv2.RANSAC)
    return H


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

    if VERBOSE:
        print 'area of head in fore img: ' + str(area_fore)
        print 'area of head in back img: ' + str(area_back)

    # NOTE: back_image is actually the foreground, fore_image is background
    if area_back < area_fore:
        return True
    else:
        return False


def blend_cropped_image(background_img, input_img):
    """
    Blends and merges the cropped image with the background
    :param background_img: real background unlike the rest of the code 
    :param input_img: input image
    :return: 
    """
    height, width, channels = input_img.shape
    cropped_img_binary = np.zeros((height, width, 1), np.uint8)

    # create binary mask
    cropped_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    cv2.threshold(cropped_img_gray, 0, 255, cv2.THRESH_BINARY, \
                  cropped_img_binary)

    # display binary mask
    concat_image = np.concatenate((imresize(cropped_img_gray, scale_factor), \
                                   imresize(cropped_img_binary, scale_factor)),
                                  axis=1)
    # show('binarized cropped image', concat_image)
    # create a cut and paste version
    merged_img = np.uint8((255 - cropped_img_binary) / 255) * background_img + \
                 input_img
    element_sizes = [(5, 0)]
    outer_mask = cropped_img_binary
    for erosion_size, fore_coeff in element_sizes:
        element_size = (erosion_size, erosion_size)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, element_size)
        inner_mask = np.zeros((height, width, 1), np.uint8)
        cv2.erode(outer_mask, element, dst=inner_mask)

        mask = outer_mask - inner_mask
        blended_section = np.uint8(fore_coeff * mask / 255 * input_img + \
                                   (
                                       1 - fore_coeff) * mask / 255 * background_img)

        inverse_mask = 255 - mask
        inverse_mask = inverse_mask / 255  # convert to 1s and 0s
        merged_img = inverse_mask * merged_img + blended_section

        outer_mask = inner_mask

    rows, cols, channels = inner_mask.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if inner_mask[i, j] == np.uint8(255):
                bottom_row = i
                break
    print 'bottom row: ' + str(bottom_row)
    # show('final mask', imresize(inner_mask,scalefactor))
    merged_img = merged_img[0:bottom_row, 0:cols]
    # show('final result', imresize(merged_img,scalefactor))
    return merged_img

def get_result_image_from_homography(fore_image, back_image, matches, kp_fore,
                                     kp_back,
                                     col_start, col_end, left_box, ap_H=False):
    H = []
    if ap_H:
        H, mask = calculate_homography(matches, kp_fore, kp_back)
        warped_back = cv2.warpPerspective(back_image, H, (
            fore_image.shape[:2][1], fore_image.shape[:2][0]))
    else:
        warped_back = back_image
    if left_box:  # figure out which is to the left of the other
        warped_back, fore_image = fore_image, warped_back
    blended_image = alpha_blend(fore_image, warped_back, col_start, col_end)
    if ap_H:
        blended_image = crop_image(blended_image, H)
    return blended_image


def main(img_fg, img_bg, res_fname):
    # set parameters
    # read images
    fore_image = cv2.imread(img_fg)
    back_image = cv2.imread(img_bg)
    # color correction
    # fore_image, back_image = colorcorrect(fore_image, back_image)
    # find body bounding box
    bb_fore, bb_fore_face = find_bounding_box(fore_image)
    bb_back, bb_back_face = find_bounding_box(back_image)

    # swap fore and back images if necessary (fore has bigger body)
    if swap_fore_back(bb_fore_face, bb_back_face):
        tmp = fore_image  # due to shallow copy
        fore_image = back_image
        back_image = tmp

        tmp = bb_fore_face
        bb_fore_face = bb_back_face
        bb_back_face = tmp

        tmp = bb_fore  # due to shallow copy
        bb_fore = bb_back
        bb_back = tmp
    # show_bounding_boxes(fore_image, back_image,bb_fore,bb_fore_face, bb_back, bb_back_face)
    rows, cols, channels = fore_image.shape
    far, col_start, col_end, left_box = bounding_box_far(bb_fore, bb_back, cols)
    if far == -1:
        if VERBOSE:
            print 'Images are not proper'
        raise ValueError

    if far == 1:
        if VERBOSE:
            print 'Image subjects are far apart, can use panorama flow'
        # treat as a 2 picture panoramic
        matches, kp_fore, kp_back = find_keypoint_matches(fore_image,
                                                          back_image)
        matches = prune_keypoint_matches_by_distance(matches)
        matches = prune_keypoint_matches_by_bounding_box(matches, kp_fore,
                                                         bb_fore, kp_back,
                                                         bb_back)
        # display_keypoint_matches(fore_image, back_image, kp_fore, kp_back, matches)

        res_img = get_result_image_from_homography(fore_image, back_image,
                                                   matches, kp_fore,
                                                   kp_back, col_start, col_end,
                                                   left_box)
        # show('blended image', imresize(blended_image,scalefactor))
        # cv2.imwrite(res_fname, cropped_image)
        cv2.imwrite(res_fname, res_img)
    else:
        if VERBOSE:
            print 'Image subjects too close together, need to do segmentation.'
        matches, kp_fore, kp_back = find_keypoint_matches(fore_image,
                                                          back_image)
        # display_keypoints(fore_image, back_image, kp_fore, kp_back)
        matches = prune_keypoint_matches_by_distance(matches)
        matches = prune_keypoint_matches_by_bounding_box(matches, kp_fore,
                                                         bb_fore, kp_back,
                                                         bb_back)
        # display_keypoint_matches(fore_image, back_image, kp_fore, kp_back, matches)
        H, mask = calculate_homography(matches, kp_fore, kp_back)
        warped_back = cv2.warpPerspective(back_image, H, (
            fore_image.shape[:2][1], fore_image.shape[:2][0]))
        warped_back_cut = grab_cut.grab_cut(warped_back)
        blended_image = blend_cropped_image(fore_image, warped_back_cut)
        # show('blended image', imresize(blended_image, scalefactor))
        cv2.imwrite(res_fname, imresize(blended_image, scale_factor))
