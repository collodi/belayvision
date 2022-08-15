import torch
import numpy as np
import cv2
from sympy import Point, Polygon, Segment, N

def rect_to_polygon(r):
	p1 = Point(r[0], r[1])
	p2 = Point(r[2], r[1])
	p3 = Point(r[2], r[3])
	p4 = Point(r[0], r[3])
	return Polygon(p1, p2, p3, p4)

# for altitude threshold (predetermined per camera)
safe = map(Point, [(0, 1400), (1079, 1400), (1079, 1920), (0, 1920)])
safe = Polygon(*safe)

# autobelay box (predetermined per camera per autobelay)
ab = rect_to_polygon([537, 145, 616, 230])

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
imgs = [cv2.imread(f'./imgs/{i}.jpg')[..., ::-1] for i in range(15)]

def left_line_area_segment(human):
	l1 = Segment(ab.vertices[3], human.vertices[0]) # bigger (cuz of reverse y in image)
	l2 = Segment(ab.vertices[3], human.vertices[3])
	# if slopes are same, should return l1 (top left)
	return l1 if l1.slope >= l2.slope else l2

def right_line_area_segment(human):
	l1 = Segment(ab.vertices[2], human.vertices[1]) # smaller (cuz of reverse y in image)
	l2 = Segment(ab.vertices[2], human.vertices[2])
	# if slopes are same, should return l1 (top right)
	return l1 if l1.slope <= l2.slope else l2

def filter_image_with_polygon(img, poly):
	poly_pts = np.array(poly.vertices)
	poly_pts = poly_pts.astype(int)

	mask = np.zeros(img.shape[0:2], dtype=np.uint8)
	cv2.drawContours(mask, [poly_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
	return cv2.bitwise_and(img, img, mask = mask)

def get_humans(xyxyn):
	humans = xyxyn[(xyxyn[:, 5] == 0.) & (xyxyn[:, 4] > .6)]
	return [rect_to_polygon(h) for h in humans[:, :4]]

def above_height_threshold(humans):
	return [h for h in humans if len(safe.intersection(h)) == 0]

def get_line_area_polygon(human):
	left = left_line_area_segment(human)
	right = right_line_area_segment(human)

	return Polygon(left.points[0], right.points[0], right.points[1], left.points[1])

def line_on_boundary(poly, ln):
	ln = Segment(Point(ln[:2]), Point(ln[2:]))

	smaller_poly = poly.scale(0.9)
	return len(smaller_poly.intersection(ln)) == 0

def detect_line(img, human):
	poly = get_line_area_polygon(human)
	line_area = filter_image_with_polygon(img, poly)

	# height of line area
	closer_bottom = min(poly.vertices[2][1], poly.vertices[3][1])
	h = abs(poly.vertices[1][1] - closer_bottom)

	threshold = int(h * 0.50)
	minlen = h * 0.70
	gap = h * 0.1

	canny = cv2.Canny(line_area, 50, 100, None, 3)
	linesP = cv2.HoughLinesP(canny, 1, np.pi / 360, threshold, None, minlen, gap)

	if linesP is not None:
		for i in range(0, len(linesP)):
			l = linesP[i][0]
			if line_on_boundary(poly, l):
				continue

			cv2.line(line_area, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

	cv2.imshow('lines', line_area)
	cv2.waitKey(0)

def main():
	results = model(imgs)
	xyxys = results.xyxy

	for i in range(15):
		print(f'=== img {i}')

		humans = get_humans(xyxys[i])
		high_humans = above_height_threshold(humans)

		print(f'{len(humans)} humans')
		print(f'{len(high_humans)} high humans')

		if len(high_humans) > 0:
			detect_line(imgs[i], high_humans[0])

if __name__ == '__main__':
	main()
