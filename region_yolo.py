import cv2
import numpy as np
import torch
import time
import torchvision
import torch.nn.functional as F
import kornia
import pandas as pd

from darknet import Darknet


class RegionYolo:

	def __init__(self, args):
		super(RegionYolo, self).__init__()
		
		self.args = args
		cuda = torch.cuda.is_available()
		self.device = (
			torch.device('cuda:0' if cuda else 'cpu')
		)
		self.dtype = (
			torch.cuda.FloatTensor if cuda else torch.FloatTensor
		)
		self.class_names = self._load_class_names(self.args.objnames)
		
		self.m = Darknet(self.args.cfg_yolo).to(self.device)
		self.m.load_weights(self.args.weights_yolo)
		self.m.eval()
		self.nms_thresh = 0.6
		self.iou_tresh = 0.4

		self.map_name = {}
		for n, i in enumerate(self.class_names):
			self.map_name[i] = n

		if cuda:
			id_ = torch.cuda.current_device()
			print(f"\nGPU ID: {id_}")
			print(f"GPU Name: {torch.cuda.get_device_name(id_)}\n")
			assert next(self.m.parameters()).is_cuda, 'Model Not on GPU'

			print('GPU WARMUP...')
			x = torch.rand(1, 3, 416, 416).cuda()
			torch.cuda.synchronize()
			for i in range(10):
				torch.cuda.synchronize()
				a = time.perf_counter()
				y = self.m(x, 0.6)
				torch.cuda.synchronize()
				b = time.perf_counter()
				print(f'batch GPU {b - a}s', end='\r')
			print('')
		else:
			print(f"\nRunning on {self.device}\n")

		self.time_proceesing = []
		self.time_nms = []


	def convert_coordinates(self, image, box):

		w = image.shape[1]
		h = image.shape[0]

		x1 = int(np.around((box[0] - box[2]/2.0) * w))
		y1 = int(np.around((box[1] - box[3]/2.0) * h))
		x2 = int(np.around((box[0] + box[2]/2.0) * w))
		y2 = int(np.around((box[1] + box[3]/2.0) * h))
		coord = np.array([x1, y1, x2, y2])

		return coord

	def detect_boxes(self, image):
		
		processed_image = self._process_img(image)
		boxes = self._detect_objects(processed_image)
		# boxes = (
		# 		[l for l in boxes ]#if l[6] == self.map_name["artefact"]]
		# 	)
		print('Done.\n')
		return np.array(boxes)

	def _process_img(self, image):
		# resized_image = cv2.resize(image, (self.m.width, self.m.height))
		img = kornia.image_to_tensor(image).to(self.device).type(self.dtype)
		img = kornia.bgr_to_rgb(img)
		img = img.div(255.0).unsqueeze(0)

		resized_image = F.interpolate(
			img,
			size=(self.m.width, self.m.height),
			mode='nearest'		)
		return resized_image

	def _detect_objects(self, img):
		print('Detecting...')
		start = time.time()
		with torch.no_grad():
			list_boxes = self.m(img, self.nms_thresh)
		end = time.time()

		boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
		
		print('Performing NMS...')
		start2 = time.time()
		boxes = self._nms(boxes)
		end2 = time.time()

		self.time_proceesing.append(round(end - start, 4))
		self.time_nms.append(round(end2 - start2, 4))

		return boxes


	def _nms(self, boxes):

		if len(boxes) == 0:
			return boxes  

		det_confs = torch.zeros(len(boxes)).to(self.device).type(self.dtype)
		
		for i in range(len(boxes)):
			det_confs[i] = boxes[i][4]

		_,sortIds = torch.sort(det_confs, descending=True)		

		boxes_tensor = torch.FloatTensor(boxes).to(self.device).type(self.dtype)
		boxes_tensor = boxes_tensor[sortIds]

		cloned_tensor = boxes_tensor.clone()
		cloned_tensor = cloned_tensor.to(self.device).type(self.dtype)

		# transform w and h to x2, y2
		cloned_tensor[:, 2] = cloned_tensor[:, 0] + cloned_tensor[:, 2]
		cloned_tensor[:, 3] = cloned_tensor[:, 1] + cloned_tensor[:, 3]

		boxes = cloned_tensor[:, :4]
		scores = cloned_tensor[:, 4]

		indices = torchvision.ops.nms(boxes, scores, self.iou_tresh)
		boxes_tensor = boxes_tensor[indices]

		return boxes_tensor.tolist()


	def _load_class_names(self, namesfile):
		class_names = []
		with open(namesfile, 'r') as fp:
			lines = fp.readlines()
		
		for line in lines:
			line = line.rstrip()
			class_names.append(line)  
		return class_names

	def expand_bbox(self, left, right, top, bottom, img_width, img_height):
		width = right-left
		height = bottom-top
		ratio = 0.04
		new_left = np.clip(left-ratio*width,0,img_width)
		new_right = np.clip(right+ratio*width,0,img_width)
		new_top = np.clip(top-ratio*height,0,img_height)
		new_bottom = np.clip(bottom+ratio*height,0,img_height)

		return np.array([int(new_left), int(top), int(new_right), int(bottom)])
		