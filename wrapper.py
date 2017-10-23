from tqdm import tqdm
import video_maker
from main import main

import cv2

out_dir = 'output'
fra_dir = 'frames/'


def extract_frames(video):
    """
    Extract frames from a given video and save them as jpg files
    :param video: name of input video file 
    :return: number of frames in the given video
    """
    vidcap = cv2.VideoCapture('videos/' + video + '.mp4')
    count = 1
    while True:
        success, image = vidcap.read()
        if success:
            cv2.imwrite(fra_dir + video + '_%d.jpg' % count, image)
            count += 1
        else:
            break
    return count - 1


cnt1 = extract_frames('video_fg')
print('fg done')
cnt2 = extract_frames('video_bg')
print('bg done')
res = []
cnt = 0
fl = 0
width = 480
height = 640
for i in tqdm(range(1, min(cnt1, cnt2))):
    j = str(i)
    print(j)
    try:
        cnt += 1
        main(img_bg=fra_dir + '/video_bg_' + j + '.jpg',
             img_fg=fra_dir + '/video_fg_' + j + '.jpg',
             res_fname=out_dir + '/result_' + str(cnt) + '.jpg')
    except Exception as e:
        cnt -= 1
        fl += 1
        pass
print 'Frames lost:', fl

video_maker.main('output', 'jpg', 'result.avi')
