import sys
import numpy as np
import cv2
import argparse
import os
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser(description='video frame extractor')
parser.add_argument('--video', type=str, required=True, help='input video')
parser.add_argument('--start', type=float, default=0, help='start position in secs for subclip')
parser.add_argument('--end', type=float, default=0, help='end position in secs for subclip')
parser.add_argument('--output_name', type=str, default="frame", help='name for output files')
parser.add_argument('--output_dir', type=str, default="frames", help='path for putput frames')
parser.add_argument('--skip', type=int, default=0, help='skip n frames fo reduce output')
parser.add_argument('--png', action='store_true', help='store as png (default is jpeg)')
parser.add_argument('--square', type=int, default=0, help='')
opt = parser.parse_args()

# directory for storing the frames
# TBD create directory if it does not exist
output_dir = opt.output_dir

# input video file name from the command line
# usage: python v2frames.py videofile.mp4 
vname = opt.video

name = opt.output_name

clip = VideoFileClip(vname) 

if opt.start > 0 and opt.end > 0:
    clip = clip.subclip(opt.start, opt.end)

count = 1
skip = 0
if opt.png:
    ext = ".png"
else:
    ext = ".jpg"



for frame in clip.iter_frames():
    if skip > 0:
        skip = skip - 1
        continue
    img = np.array(frame, dtype=np.uint8)
    print(count)
    # change rgb to bgr as we are using opencv to write the frames to disk 
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if opt.square > 0:
        backgr = np.full((opt.square, opt.square, 3), 255, np.uint8) #.fill(255)
        print(backgr)
        h, w = img_out.shape[:2]
        print(h,w)
        r = opt.square / w
        h = int(r * h)
        w = int(r * w)
        img_ovr = cv2.resize(img_out, (w, h))
        print(img_ovr.shape)
        h1 = int((opt.square/2) - h/2)
        print(h1, h, w)
        backgr[h1:h1+h, :, :] = img_ovr
        img_out = backgr 
    cv2.imwrite(os.path.join(output_dir, name + str(count) + ext), img_out)
    count = count + 1
    if opt.skip > 0:
        skip = opt.skip

