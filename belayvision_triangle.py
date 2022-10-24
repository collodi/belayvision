import torch
import numpy as np
import cv2
from sympy import Point, Polygon, Segment, N
import skvideo.io

def show_image(img):
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image', img)
	cv2.resizeWindow('image', 675, 1200)
	cv2.waitKey(0)

def rect_to_polygon(r):
	p1 = Point(r[0], r[1])
	p2 = Point(r[2], r[1])
	p3 = Point(r[2], r[3])
	p4 = Point(r[0], r[3])
	return Polygon(p1, p2, p3, p4)

def poly_to_int_array(poly):
	poly_pts = np.array(poly.vertices)
	return poly_pts.astype(int)

def filter_image_with_polygon(img, poly):
	poly_pts = np.array(poly.vertices)
	poly_pts = poly_pts.astype(int)

	mask = np.zeros(img.shape[0:2], dtype=np.uint8)
	cv2.drawContours(mask, [poly_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
	return cv2.bitwise_and(img, img, mask = mask)

def crop_image_with_polygon(img, poly):
	poly_pts = np.array(poly.vertices)
	poly_pts = poly_pts.astype(int)

	x, y, w, h = cv2.boundingRect(poly_pts)
	return img[y:y+h, x:x+w]

# for altitude threshold (predetermined per camera)
safe = map(Point, [(0, 1400), (1079, 1400), (1079, 1920), (0, 1920)])
safe = Polygon(*safe)

# autobelay box (predetermined per camera per autobelay)
ab = rect_to_polygon([537, 145, 616, 230]) # xyxy

# AB triangle box (predetermined per camera per autobelay)
triangle = map(Point, [(506, 1396), (647, 1727), (352, 1740)])
triangle = Polygon(*triangle)

# AB triangle up comparison pic
triangle_img = cv2.imread('triangle.png')
triangle_img = cv2.cvtColor(triangle_img, cv2.COLOR_BGR2LAB)

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
imgs = [cv2.imread(f'./imgs/{i}.jpg') for i in range(15)]
imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

def get_humans(xyxy):
	humans = xyxy[(xyxy[:, 5] == 0.) & (xyxy[:, 4] > .6)]
	return [rect_to_polygon(h) for h in humans[:, :4]]

def above_height_threshold(humans):
	return [h for h in humans if len(safe.intersection(h)) == 0]

def is_triangle_up(img):
	tr_img = filter_image_with_polygon(img, triangle)
	tr_img = crop_image_with_polygon(tr_img, triangle)

	tr_img = cv2.cvtColor(tr_img, cv2.COLOR_RGB2LAB)

	diff = abs(triangle_img - tr_img) > 10
	diff = diff.astype(np.int8)

	# average difference (0/1) per each pixel in the triangle
	distance = N(diff.sum() / triangle.area)

	print(distance)

	return distance < 2.

def is_image_dangerous(img, xyxy):
	humans = get_humans(xyxy)
	high_humans = above_height_threshold(humans)

	if len(high_humans) == 0:
		return False

	if len(high_humans) > 1:
		return True

	return is_triangle_up(img)

def visualize_img(img, xyxy):
	humans = get_humans(xyxy)
	high_humans = above_height_threshold(humans)
	t_up = is_triangle_up(img)

#	print(f'{len(humans)} humans')
#	print(f'{len(high_humans)} high humans')
#	print(f'triangle up? {t_up}')

	# leave the original image alone in case we need to do more processing
	img = img.copy()

	# altitude threshold
	pts = poly_to_int_array(safe)
	cv2.line(img, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 3)

	# box every human in image
	for h in humans:
		pts = poly_to_int_array(h)
		cv2.rectangle(img, tuple(pts[0]), tuple(pts[2]), (0, 255, 0), 3)

	# high humans have red bottom border
	for h in high_humans:
		pts = poly_to_int_array(h)
		cv2.line(img, tuple(pts[2]), tuple(pts[3]), (255, 0, 0), 3)

	if t_up:
		pts = poly_to_int_array(triangle).reshape(-1, 1, 2)
		cv2.polylines(img, [pts], True, (0, 255, 0), 3)

	return img

def display_video(vid):
	cap = cv2.VideoCapture(vid)
	cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.)

	if not cap.isOpened():
		print(f'Error opening the video: {vid}')
		return

	frame_cnt = 0
	calc_rate = int(cap.get(cv2.CAP_PROP_FPS)) // 4

	cv2.namedWindow('video', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('video', 675, 1200)

	while cap.isOpened():
		ret, frame = cap.read()

		if not ret:
			break

		frame_cnt += 1
		if frame_cnt != calc_rate:
			continue
		else:
			frame_cnt = 0

		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		results = model(frame)
		vis = visualize_img(frame, results.xyxy[0])
		cv2.imshow('video', vis[..., ::-1])

		if cv2.waitKey(1) & 0xff == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def save_video(vid):
	cap = cv2.VideoCapture(vid)
	cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.)

	if not cap.isOpened():
		print(f'Error opening the video: {vid}')
		return

	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

	frame_cnt = 0
	calc_rate = 1
	#calc_rate = frame_rate // 4

	writer = skvideo.io.FFmpegWriter('out.mp4',
			inputdict={'-r': f'{frame_rate // calc_rate}'}
	)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		frame_cnt += 1
		if frame_cnt != calc_rate:
			continue
		else:
			frame_cnt = 0

		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		results = model(frame)
		vis = visualize_img(frame, results.xyxy[0])

		writer.writeFrame(vis)

	writer.close()
	cap.release()

def main():
	#	tr_img = filter_image_with_polygon(imgs[0], triangle)
	#	tr_img = crop_image_with_polygon(tr_img, triangle)
	#
	#	tr_img = tr_img[..., ::-1]
	#
	#	cv2.imwrite('triangle.png', tr_img)
	#	cv2.imshow('triangle', tr_img)
	#	cv2.waitKey(0)
	#	return

	save_video('vids/0.mp4')
	return

	results = model(imgs)
	xyxys = results.xyxy

	for i in range(15):
		print(f'=== img {i}')

		danger = is_image_dangerous(imgs[i], xyxys[i])
		print(f'dangerous: {danger}')

		vis = visualize_img(imgs[i], xyxys[i])
		show_image(vis[..., ::-1])

if __name__ == '__main__':
	main()
