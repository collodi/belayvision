import torch
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

# AB triangle box (predetermined per camera per autobelay)
triangle = [(506, 1396), (647, 1727), (352, 1740)]

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
imgs = [cv2.imread(f'./imgs/{i}.jpg')[..., ::-1] for i in range(15)]

def get_humans(xyxyn):
	humans = xyxyn[(xyxyn[:, 5] == 0.) & (xyxyn[:, 4] > .6)]
	return [rect_to_polygon(h) for h in humans[:, :4]]

def above_height_threshold(humans):
	return [h for h in humans if len(safe.intersection(h)) == 0]

def main():
	results = model(imgs)
	xyxys = results.xyxy

	for i in range(15):
		print(f'=== img {i}')

		humans = get_humans(xyxys[i])
		high_humans = above_height_threshold(humans)

		print(f'{len(humans)} humans')
		print(f'{len(high_humans)} high humans')

		# TODO detect triangle

if __name__ == '__main__':
	main()
