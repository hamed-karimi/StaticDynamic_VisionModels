import cv2
import os
from PIL import Image
import sys
from datetime import datetime

def FrameCapture(video_path):
    vid = cv2.VideoCapture(video_path)
    ind = 0
    frames = []
    while(vid.isOpened()):
        success, img = vid.read()
        if success == False:
            break
        frames.append(img)
        
    return frames


n_jobs = 48
shift = int(sys.argv[1])
video_inds = []

st = datetime.now()
root='/data/karimike/Documents/ActionRecognition-2024/Kinetics-400/kinetics-dataset/k400'
modes = ['train', 'val', 'test']
for mode in modes:
    dataset = os.path.join(root, mode)
    output_dir = os.path.join(root, '{0}_frames'.format(mode))
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    videos = os.listdir(dataset)
    videos = sorted(videos)
    video_inds = [i*n_jobs + shift for i in range(len(videos)//n_jobs)]
    
    for v_id in video_inds:
        video_name = videos[v_id]
        if video_name.endswith('.mp4'):
            try:
                frames = FrameCapture(os.path.join(dataset, video_name))
                frames_dir = os.path.join(output_dir, video_name.split('.')[0])
                if not os.path.exists(frames_dir):
                    os.mkdir(frames_dir)
                    for f_i, frame in enumerate(frames):
                        cv2.imwrite(os.path.join(frames_dir, '{0}.jpg'.format(f_i)), frame)
            except:
                print('error in video with index', v_id, 'name', video_name)
    break
print(datetime.now() - st)