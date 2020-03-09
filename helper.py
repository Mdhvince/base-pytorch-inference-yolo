import cv2
import numpy as np
import imutils
import ffmpeg
import sys
import os
import time
from imutils.video import FPS


def draw_poses(image, poses, dim):

	poses_draw = poses.astype(int)
	pairs = np.array([ [0, 1], [1, 2], [3, 4], [4, 5],
		[6, 7], [7, 8], [9, 10], [10, 11] ])

	for e in pairs:
		cv2.line(image, tuple(poses_draw[e[0]]),
			tuple(poses_draw[e[1]]), (255,255,255), 3, lineType=cv2.LINE_AA)
		
	for j, i in enumerate(poses_draw):
		cv2.circle(image, tuple(poses_draw[j]), 4, (0,255,0), -1, lineType=cv2.LINE_AA)

	return image


def set_window_pos(name):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 0, 0)


def init_capture_video(args):
	cap = cv2.VideoCapture(args.videopath)
	time.sleep(2.0)
	rotateCode = None
	rotateCode = check_rotation(args.videopath)
	tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = FPS().start()

	return cap, tot_frames, fps, rotateCode

def init_capture_camera(val):
	cap = cv2.VideoCapture(val)
	fps = FPS().start()

	return cap, fps


def safe_close(fps, cap):
	fps.stop()
	print(f"\nApprox FPS during Inference: {fps.fps()}")
	cap.release()
	cv2.destroyAllWindows()


def check_rotation(path_video_file):
	val=0
	# this returns meta-data of the video file in form of a dictionary
	meta_dict = ffmpeg.probe(path_video_file)
	# from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
	# we are looking for
	rotate_code = None
	rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
	val = round(int(rotate) / 90.0) * 90
	return val

def correct_rotation(frame, rotateCode):
	corrected = imutils.rotate_bound(frame, rotateCode)
	return corrected

def rescale_out(poses, dim):
	poses[:, -1] += dim[1]
	poses[:, :1] += dim[0]
	return poses











