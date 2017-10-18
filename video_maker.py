import functools
import cv2
import os

def n(s):
    i = s.split('.')[0].split('_')[1]
    return int(i)

def comp(s1, s2):
    return n(s1) - n(s2)

def main(dir_path, ext, output):
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    images = sorted(images, key=functools.cmp_to_key(comp))
    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20, (width, height))

    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))

if __name__ == "__main__":
    main('output','jpg','result.avi')