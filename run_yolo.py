import cv2
import numpy as np
import torch
import urllib.request as urlb 
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from region_yolo import RegionYolo
import helper


def parserf():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--videopath',
		type=str, default=''
	)
	parser.add_argument(
		'--imgpath',
		type=str, default=''
	)
	parser.add_argument(
		'--objnames',
		type=str
	)
	parser.add_argument(
		'--cfg_yolo',
		type=str
	)
	parser.add_argument(
		'--weights_yolo',
		type=str
	)
	return parser.parse_args()

def url_to_image(url):
	resp = urlb.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

if __name__ == '__main__':
	args = parserf()
	detector = RegionYolo(args)
	color = (0, 255, 0)

	if args.imgpath != '':
		image = url_to_image(args.imgpath)
		boxes = detector.detect_boxes(image)

		print(f'Nb detections: {len(boxes)}')

		_, ax = plt.subplots(1)
		ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

		for box in boxes:
			coords = detector.convert_coordinates(image, box)
			text = f'{detector.class_names[int(box[6])]}: {round(box[5], 2)}'
			rect = patches.Rectangle(
				(coords[0], coords[1]),
				coords[2]-coords[0], coords[3]-coords[1],
				linewidth=1,
				edgecolor='r',
				facecolor='none'
			)
			ax.add_patch(rect)
			ax.text(
				coords[0], coords[1],
				text,
				bbox=dict(facecolor='white', alpha=0.5))

		plt.axis('off')
		plt.show()
		print(f"Mean Time detection per frame: {np.mean(np.array(detector.time_proceesing))}s")
		print(f"Mean Time NMS per frame: {np.mean(np.array(detector.time_nms))}s")
		
	else:
		if args.videopath != '':
			cap, tot_frames, fps, rotateCode = helper.init_capture_video(args)
		else:
			cap, fps = helper.init_capture_camera(0)

		name_w = 'frame'
		helper.set_window_pos(name_w)
		while (cap.isOpened()):
			ret_val , frame = cap.read()
			if not ret_val: break

			if args.videopath != '':
				if rotateCode is not None:
					frame = helper.correct_rotation(frame, rotateCode)

			boxes = detector.detect_boxes(frame)

			for box in boxes:
				coords = detector.convert_coordinates(frame, box)
				cv2.rectangle(
					frame,
					(coords[0], coords[1]),
					(coords[2], coords[3]),
					color, 2)

				text = f'{detector.class_names[int(box[6])]}: {round(box[5], 2)}'
				cv2.putText(frame, text, (coords[0], coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 2)

				cv2.imshow(name_w, frame)

			if cv2.waitKey(1) & 0xFF == ord('q'): break
			fps.update()

		print(f"Mean Time detection per frame: {np.mean(np.array(detector.time_proceesing))}s")
		print(f"Mean Time NMS per frame: {np.mean(np.array(detector.time_nms))}s")
		helper.safe_close(fps, cap)


